import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
import os

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from transformers import RobertaModel

from .layers.lora import LinearWithLoRA

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


class SpatialImageLanguageAttention(nn.Module):
    """ Spatial Image-Language Attention Module """
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x (visual): (B, H*W, v_in_channels)
        # l (language): (B, l_in_channels, N_l)
        # l_mask: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Key: generated from language features
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Query: generated from visual features
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Value: generated from language features
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Output projection layer
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x (visual): (B, H*W, v_in_channels)
        # l (language): (B, N_l, l_in_channels)
        # l_mask: (B, N_l)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, v_in_channels, H*W)
        l = l.permute(0, 2, 1)  # (B, l_in_channels, N_l)
        l_mask = l_mask.permute(0, 2, 1)  # (B, 1, N_l)

        # 1. Generate Query, Key, Value
        query = self.f_query(x)  # (B, key_channels, H*W)
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, value_channels, N_l)

        # 2. Apply language mask to ignore padding tokens
        key = key * l_mask
        value = value * l_mask
        n_l = value.size(-1)

        # 3. Reshape for multi-head attention
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, N_l)

        # 4. Compute attention scores (similarity map)
        sim_map = torch.matmul(query, key)  # (B, num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # Scale

        # 5. Apply language mask again, set padding positions to very small values
        sim_map = sim_map + (1e4*l_mask - 1e4)
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, H*W, N_l)

        # 6. Weighted sum of Value based on attention scores
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)

        # 7. Final output projection
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out


class PWAM(nn.Module):
    """ Pixel-Word Alignment Module """
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # Projection layer for visual features
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        # Core image-language spatial attention module
        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,
                                                            l_in_channels,
                                                            key_channels,
                                                            value_channels,
                                                            out_channels=value_channels,
                                                            num_heads=num_heads)

        # Projection layer for fused features
        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # x: (B, H*W, dim)
        # 1. Project visual features
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        # 2. Compute language-guided features
        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)
        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        # 3. Element-wise multiplication for fusion
        mm = torch.mul(vis, lang)
        # 4. Project fused features
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm

class LanguageVisionAttention(nn.Module):
    """ Language-Vision Spatial Attention Module (text absorbs visual) """
    def __init__(self, l_in_channels, v_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(LanguageVisionAttention, self).__init__()
        # l (language): (B, N_l, l_in_channels) -> Query
        # x (visual): (B, H*W, v_in_channels) -> Key, Value
        self.l_in_channels = l_in_channels
        self.v_in_channels = v_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Query: generated from language features
        self.f_query = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )
        # Key: generated from visual features
        self.f_key = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )
        # Value: generated from visual features
        self.f_value = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        # Output projection layer
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, l, x):
        B, N_l = l.size(0), l.size(1)
        HW = x.size(1)

        l = l.permute(0, 2, 1)  # (B, l_in_channels, N_l)
        x = x.permute(0, 2, 1)  # (B, v_in_channels, H*W)

        query = self.f_query(l).permute(0, 2, 1)  # (B, N_l, key_channels)
        key = self.f_key(x)    # (B, key_channels, H*W)
        value = self.f_value(x)  # (B, value_channels, H*W)

        query = query.reshape(B, N_l, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, HW)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, HW)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, N_l, self.value_channels)

        out = out.permute(0, 2, 1)
        out = self.W(out)
        out = out.permute(0, 2, 1)

        return out

class LVAM(nn.Module):
    """ Language-Vision Alignment Module """
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(LVAM, self).__init__()
        self.lang_vis_att = LanguageVisionAttention(
            l_in_channels, v_in_channels, key_channels, value_channels,
            out_channels=value_channels, num_heads=num_heads
        )
        self.project_mm = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, l, x):
        # l: (B, N_l, dim), x: (B, H*W, dim)
        vis_guided_l = self.lang_vis_att(l, x)
        mm = self.project_mm(vis_guided_l)
        return mm

class Pi3(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            pretrained_model_name_or_path=None,
            use_referring_segmentation=False,
            text_model_name='./ckpts/roberta-base',
            freeze_visual_modules=False,
            use_masked_attn=True,
            num_multimodal_layers=12,
            use_pretrained_weights=True,
            use_lora=False,
            use_lang_vision_fusion=False,
            lvam_interval=1,
            reinject_visual_features=False,
            reinject_interval=2,
            multimodal_layer_selection='back',
            fusion_mode='pwa_only',
            use_controlnet_injection=True,
            controlnet_injection_interval=1,
            pwam_injection_mode='all',
            use_simple_cross_attention=False
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token
        self.use_masked_attn = use_masked_attn

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)
        

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # --------------------------------
        #   Referring Segmentation Module
        # --------------------------------
        self.use_referring_segmentation = use_referring_segmentation
        if self.use_referring_segmentation:
            self.use_simple_cross_attention = use_simple_cross_attention
            self.pwam_injection_mode = pwam_injection_mode
            self.use_pretrained_weights = use_pretrained_weights
            self.use_lora = use_lora
            self.use_lang_vision_fusion = use_lang_vision_fusion
            self.lvam_interval = lvam_interval
            self.reinject_visual_features = reinject_visual_features
            self.reinject_interval = reinject_interval
            self.use_controlnet_injection = use_controlnet_injection
            self.controlnet_injection_interval = controlnet_injection_interval
            self.multimodal_layer_selection = multimodal_layer_selection
            self.fusion_mode = fusion_mode
            
            # 1. Text encoder
            self.text_encoder = RobertaModel.from_pretrained(text_model_name, add_pooling_layer=False)
            roberta_dim = self.text_encoder.config.hidden_size # Usually 768

            # 2. Text projection layer
            self.text_proj = nn.Linear(roberta_dim, self.dec_embed_dim)

            self.dec_num_heads = dec_num_heads
            # 3. Multimodal interaction module - extend to multiple layers
            num_fusion_layers =  num_multimodal_layers

            if self.use_simple_cross_attention:
                self.simple_cross_attention = nn.TransformerDecoderLayer(
                    d_model=self.dec_embed_dim,
                    nhead=self.dec_num_heads,
                    dim_feedforward=self.dec_embed_dim * 4,
                    batch_first=True
                )

            if multimodal_layer_selection == 'front':
                layer_indices = list(range(num_fusion_layers))
            elif multimodal_layer_selection == 'middle':
                start_index = (dec_depth // 2) - (num_fusion_layers // 2)
                layer_indices = list(range(start_index, start_index + num_fusion_layers))
            elif multimodal_layer_selection == 'back':
                start_index = dec_depth - num_fusion_layers
                layer_indices = list(range(start_index, dec_depth))
            elif multimodal_layer_selection == 'uniform':
                layer_indices = torch.linspace(0, dec_depth - 1, num_fusion_layers).round().long().tolist()
            else:
                raise ValueError(f"Invalid multimodal_layer_selection: {multimodal_layer_selection}")
            
            self.layer_indices = layer_indices
            self.layer_indices_map = {global_idx: local_idx for local_idx, global_idx in enumerate(self.layer_indices)}
            self.multimodal_decoder = nn.ModuleList([deepcopy(self.decoder[i]) for i in self.layer_indices])
            self.fusion_modules = nn.ModuleList([
                PWAM(
                    dim=dec_embed_dim,
                    v_in_channels=dec_embed_dim,
                    l_in_channels=dec_embed_dim,
                    key_channels=dec_embed_dim,
                    value_channels=dec_embed_dim,
                    num_heads=dec_num_heads
                ) for _ in range(num_fusion_layers)
            ])
            self.res_gate = nn.Sequential(
                nn.Linear(dec_embed_dim, dec_embed_dim, bias=False),
                nn.ReLU(),
                nn.Linear(dec_embed_dim, dec_embed_dim, bias=False),
                nn.Tanh()
            )
            if use_lang_vision_fusion or self.fusion_mode == 'interleaved':
                self.lang_fusion_modules = nn.ModuleList([
                    LVAM(
                        dim=dec_embed_dim,
                        v_in_channels=dec_embed_dim,
                        l_in_channels=dec_embed_dim,
                        key_channels=dec_embed_dim,
                        value_channels=dec_embed_dim,
                        num_heads=dec_num_heads
                    ) for _ in range(num_fusion_layers)
                ])
                self.lang_res_gate = nn.Sequential(
                    nn.Linear(dec_embed_dim, dec_embed_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(dec_embed_dim, dec_embed_dim, bias=False),
                    nn.Tanh()
                )
            
            # 5. Segmentation head
            self.mask_decoder = nn.Sequential(
                nn.Conv2d(dec_embed_dim, dec_embed_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dec_embed_dim, dec_embed_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dec_embed_dim, 1, 1)
            )

            if self.use_controlnet_injection:
                num_injections = len(self.layer_indices) // self.controlnet_injection_interval
                self.controlnet_injectors = nn.ModuleList()
                for _ in range(num_injections):
                    zero_conv = nn.Linear(dec_embed_dim, dec_embed_dim)
                    nn.init.zeros_(zero_conv.weight)
                    nn.init.zeros_(zero_conv.bias)
                    self.controlnet_injectors.append(zero_conv)

            if self.use_pretrained_weights:
                # Load multimodal_decoder weights from pretrained Pi3 model ckpts/Pi3/model.safetensors
                pi3_weight = load_file('ckpts/Pi3/model.safetensors')
                pi3_dec_weight = {k.replace('decoder.', ''):v for k,v in pi3_weight.items() if k.startswith('decoder.')}

                remapped_weights = {}
                for k, v in pi3_dec_weight.items():
                    try:
                        global_idx_str, rest_of_key = k.split('.', 1)
                        global_idx = int(global_idx_str)
                        if global_idx in self.layer_indices_map:
                            local_idx = self.layer_indices_map[global_idx]
                            new_key = f"{local_idx}.{rest_of_key}"
                            remapped_weights[new_key] = v
                    except ValueError:
                        pass

                load_result = self.multimodal_decoder.load_state_dict(remapped_weights, strict=False)
                print(f"Loading pi3 decoder to initialize multimodal_decoder. Result:")
                print(f"  Missing keys: {load_result.missing_keys}")
                print(f"  Unexpected keys: {load_result.unexpected_keys}")
            
            if self.use_lora:
                # Apply LoRA to the multimodal decoder
                for block in self.multimodal_decoder:
                    block.attn.qkv = LinearWithLoRA(block.attn.qkv, rank=16, alpha=16)
                    block.attn.proj = LinearWithLoRA(block.attn.proj, rank=16, alpha=16)
                    block.mlp.fc1 = LinearWithLoRA(block.mlp.fc1, rank=16, alpha=16)
                    block.mlp.fc2 = LinearWithLoRA(block.mlp.fc2, rank=16, alpha=16)
                
                # Freeze all parameters except LoRA
                for name, param in self.multimodal_decoder.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False


        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if load_vggt:
            # Load VGGT-1B model weights file
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')

            # 1. Load encoder weights
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            # 2. Load decoder weights
            # Decoder weights are interleaved from 'global_blocks' and 'frame_blocks' in VGGT model

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        self.train_conf = train_conf
        if train_conf:
            assert ckpt is not None

            # ----------------------
            #     Conf Decoder
            # ----------------------
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

            freeze_all_params([self.encoder, self.decoder, self.point_decoder, self.point_head, self.camera_decoder,  self.camera_head, self.register_token])
            if use_global_points:
                freeze_all_params([self.global_points_decoder, self.global_point_head])

        if freeze_visual_modules and use_referring_segmentation:
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

            print('Freezing all visual modules for referring segmentation training.')
            modules_to_freeze = [
                self.encoder,
                self.decoder,
                self.point_decoder,
                self.point_head,
                self.camera_decoder,
                self.camera_head,
                self.register_token,
                self.conf_decoder, 
                self.conf_head
            ]
            if use_global_points:
                modules_to_freeze.extend([self.global_points_decoder, self.global_point_head])
            freeze_all_params(modules_to_freeze)

        elif freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if pretrained_model_name_or_path is not None:
            pi3_ckpt = load_file(os.path.join(pretrained_model_name_or_path, 'model.safetensors'))
            self.load_state_dict(pi3_ckpt, strict=False)
            print(f'[Pi3] Load pretrained model from {pretrained_model_name_or_path}')

        if ckpt is not None:
            checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')

            del checkpoint
            torch.cuda.empty_cache()

    def decode(self, hidden, N, H, W, text_embeds=None, attention_mask=None):
        BN, hw, _ = hidden.shape
        B = BN // N
        
        layer_mask_preds = []
        if self.use_referring_segmentation and self.use_simple_cross_attention:
            # Prepare text embeds and mask for cross-attention
            text_embeds_ = text_embeds.unsqueeze(1).repeat(1, N, 1, 1).reshape(B*N, text_embeds.shape[1], -1)
            # TransformerDecoderLayer expects `True` for padding, HF mask is `1` for non-padding.
            attention_mask_ = attention_mask.unsqueeze(1).repeat(1, N, 1).reshape(B*N, attention_mask.shape[1]) == 0

            # Cross-attention
            fused_hidden = self.simple_cross_attention(tgt=hidden, memory=text_embeds_, memory_key_padding_mask=attention_mask_)

            # Predict mask
            register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
            fused_hidden_with_special = torch.cat([register_token, fused_hidden], dim=1)
            mask = self.predict_mask(fused_hidden_with_special, H, W)
            simple_mask_pred = mask.reshape(B, N, H, W)
            layer_mask_preds.append(simple_mask_pred)
            
        final_output = []
        # Invert
        # attention_mask = ~attention_mask
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        # Restore initialization of multimodal_hidden and layer_mask_preds
        multimodal_hidden = hidden.clone() if self.use_referring_segmentation else None

        final_output = []
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
                if self.use_referring_segmentation:
                    multimodal_hidden = multimodal_hidden.reshape(B*N, hw, -1)
                    text_embeds_ = text_embeds.unsqueeze(1).repeat(1, N, 1, 1).reshape(B*N, text_embeds.shape[1], -1)
                    attention_mask_ = attention_mask.unsqueeze(1).repeat(1, N, 1).reshape(B*N, attention_mask.shape[1]).unsqueeze(-1).float()
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)
                if self.use_referring_segmentation:
                    multimodal_hidden = multimodal_hidden.reshape(B, N*hw, -1)
                    text_embeds_ = text_embeds
                    attention_mask_ = attention_mask.unsqueeze(-1).float()

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if self.use_referring_segmentation and not self.use_simple_cross_attention:
                if i in self.layer_indices_map:
                    local_idx = self.layer_indices_map[i]
                    multimodal_blk = self.multimodal_decoder[local_idx]
                    if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                        multimodal_hidden = checkpoint(multimodal_blk, multimodal_hidden, xpos=pos, use_reentrant=False)
                    else:
                        multimodal_hidden = multimodal_blk(multimodal_hidden, xpos=pos)
                    multimodal_hidden_ = multimodal_hidden

                    should_inject_pwam = False
                    if self.pwam_injection_mode == 'all':
                        should_inject_pwam = True
                    elif self.pwam_injection_mode == 'even':
                        should_inject_pwam = (local_idx % 2 == 0)
                    elif self.pwam_injection_mode == 'odd':
                        should_inject_pwam = (local_idx % 2 == 1)

                    if self.fusion_mode == 'interleaved':
                        if i % 2 == 0 and should_inject_pwam: # Even layer (within-frame) -> PWAM
                            fusion_module = self.fusion_modules[local_idx]
                            x_residual = fusion_module(multimodal_hidden, text_embeds_, attention_mask_)
                            multimodal_hidden = multimodal_hidden + (self.res_gate(x_residual) * x_residual)
                        elif i % 2 != 0: # Odd layer (between-frames) -> LVAM
                            multimodal_hidden_ = multimodal_hidden.reshape(B, N*hw, -1)
                            lang_fusion_module = self.lang_fusion_modules[local_idx]
                            l_residual = lang_fusion_module(text_embeds, multimodal_hidden_)
                            text_embeds = text_embeds + (self.lang_res_gate(l_residual) * l_residual)
                    else: # 'pwa_only' mode (original behavior)
                        # PWAM Fusion
                        if should_inject_pwam:
                            fusion_module = self.fusion_modules[local_idx]
                            x_residual = fusion_module(multimodal_hidden, text_embeds_, attention_mask_)
                            
                            multimodal_hidden = multimodal_hidden + (self.res_gate(x_residual) * x_residual)

                        if self.use_lang_vision_fusion:
                            if (local_idx + 1) % self.lvam_interval == 0:
                                if i % 2 == 0:
                                    # (B*N, hw, C) -> (B, N*hw, C)
                                    multimodal_hidden_ = multimodal_hidden_.reshape(B, N*hw, -1)
                                
                                # LVAM Fusion
                                lvam_module_idx = (local_idx + 1) // self.lvam_interval - 1
                                if lvam_module_idx < len(self.lang_fusion_modules):
                                    lang_fusion_module = self.lang_fusion_modules[lvam_module_idx]
                                    l_residual = lang_fusion_module(text_embeds, multimodal_hidden_)
                                    text_embeds = text_embeds + (self.lang_res_gate(l_residual) * l_residual)

                    # Mask Prediction
                    # Unify multimodal_hidden shape to (B*N, hw, C)
                    multimodal_hidden_reshaped = multimodal_hidden.reshape(B*N, hw, -1)
                    mask_pred = self.predict_mask(multimodal_hidden_reshaped, H, W)
                    mask_pred = mask_pred.reshape(B, N, H, W)
                    layer_mask_preds.append(mask_pred)

                    # --- ControlNet-style Injection / Feature Re-injection ---
                    if self.use_controlnet_injection:
                        if (local_idx + 1) % self.controlnet_injection_interval == 0:
                            injection_idx = (local_idx + 1) // self.controlnet_injection_interval - 1
                            if injection_idx < len(self.controlnet_injectors):
                                control_signal = self.controlnet_injectors[injection_idx](hidden)
                                multimodal_hidden = multimodal_hidden + control_signal
                    elif self.reinject_visual_features and (local_idx + 1) % self.reinject_interval == 0:
                        multimodal_hidden = multimodal_hidden + hidden

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        if self.use_referring_segmentation:
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1), layer_mask_preds
        else:
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1), None

    def predict_mask(self, hidden, H, W):
        patch_h, patch_w = H // 14, W // 14
        hidden = hidden[:, self.patch_start_idx:]
        hidden = hidden.reshape(-1, patch_h, patch_w, self.dec_embed_dim).permute(0, 3, 1, 2)
        mask = self.mask_decoder(hidden)
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        return mask
    
    def forward(self, imgs, input_ids=None, attention_mask=None):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        text_embeds_proj, attention_mask_proj = None, None
        if self.use_referring_segmentation:
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state
            text_embeds_proj = self.text_proj(text_embeds)
            attention_mask_proj = attention_mask

        hidden, pos, layer_mask_preds = self.decode(hidden, N, H, W, text_embeds=text_embeds_proj, attention_mask=attention_mask_proj)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        output = {}

        if self.use_referring_segmentation:
            output['layer_referring_mask_preds'] = layer_mask_preds[:-1]
            output['referring_mask_pred'] = layer_mask_preds[-1]




        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # Global points
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                global_points = None
            
            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        output.update(dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points
        ))
        
        return output
