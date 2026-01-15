import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
import torch.nn.functional as F

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from transformers import RobertaModel

class Pi3(nn.Module, PyTorchModelHubMixin):
    """
    Core implementation of the Pi3 model.

    This is a Transformer-based multi-view geometry model that can jointly predict
    3D point clouds, camera poses, and confidence for each point from a series of input images (views).
    When use_referring_segmentation=True, the model can also accept text input and predict referring segmentation masks.
    """
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            use_referring_segmentation=True,
            text_model_name='roberta-base',
        ):
        """
        Initialize the Pi3 model.

        Args:
            pos_type (str): Type of positional encoding to use. 'rope' means Rotary Position Embedding.
            decoder_size (str): Size of decoder, options: 'small', 'base', 'large'.
            use_referring_segmentation (bool): Whether to enable referring segmentation.
            text_model_name (str): Name of the pretrained model for text encoding.
        """
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        # Use DINOv2 (ViT-L/14) as image encoder. This is a very powerful visual feature extractor.
        # The `_reg` version indicates it has a head designed for regression tasks, but here we mainly use its backbone.
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token # Remove unused mask token

        # ----------------------
        #      Positional Encoding
        # ----------------------
        # Use Rotary Position Embedding (RoPE), an advanced positional encoding technique that better captures relative positional relationships.
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # e.g., rope100
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq) # Initialize RoPE module
            self.position_getter = PositionGetter() # For generating position grid
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        # This is a custom Transformer decoder, stacked with multiple BlockRope modules.
        # It is responsible for processing and fusing multi-view features from the encoder.
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # Encoder output dimension (1024)
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
                attn_class=FlashAttentionRope, # Use FlashAttention for efficiency
                rope=self.rope # Pass RoPE module
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register Token
        # ----------------------
        # Introduce a set of learnable tokens that serve as "registers" or "summary" tokens to help the model integrate global information.
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        # A specialized Transformer decoder for decoding 3D points in local coordinate system from fused features.
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, # Input dimension is the concatenation of two-stage outputs from main decoder
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3) # Linear head, outputs (x, y, z)

        # ----------------------
        #    Confidence Decoder
        # ----------------------
        # Same structure as point decoder, but used for predicting confidence for each point.
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1) # Linear head, outputs 1D confidence

        # ----------------------
        #   Camera Pose Decoder
        # ----------------------
        # A Transformer decoder specialized for decoding camera poses.
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512) # Convert features to 4x4 pose matrix

        # --------------------------------
        #   Referring Segmentation Module
        # --------------------------------
        self.use_referring_segmentation = use_referring_segmentation
        if self.use_referring_segmentation:
            # 1. Text encoder
            self.text_encoder = RobertaModel.from_pretrained(text_model_name)
            roberta_dim = self.text_encoder.config.hidden_size # Usually 768

            # 2. Text projection layer, matching text feature dimension to visual feature dimension
            self.text_proj = nn.Linear(roberta_dim, self.dec_embed_dim)

            # 3. Multimodal interaction module (Transformer Decoder Layer)
            #    Used to fuse text features (Query) and visual features (Key/Value)
            self.multimodal_decoder = nn.TransformerDecoderLayer(
                d_model=self.dec_embed_dim,
                nhead=dec_num_heads, # Use same number of heads as main decoder
                batch_first=True # Input format is (Batch, Seq, Dim)
            )

            # 4. Segmentation head, used to generate final "segmentation token" from fused text features
            self.seg_token_head = Mlp(
                self.dec_embed_dim,
                hidden_features=self.dec_embed_dim,
                out_features=self.dec_embed_dim
            )

        # Register mean and std for image normalization (ImageNet standard)
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W):
        """
        Main decoder module responsible for fusing multi-view features.

        Args:
            hidden (torch.Tensor): Batch features from encoder output, shape (B*N, H*W, C).
            N (int): Number of views per sample.
            H (int): Image height.
            W (int): Image width.

        Returns:
            tuple: Tuple containing decoded features and positional encodings.
        """
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        # Repeat register tokens B*N times, ready to concatenate with features from each view
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate register tokens with patch features
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        # Get rotary positional encoding
        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # Don't use positional encoding for register tokens, so set their positions to 0
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        # Core decoding loop
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            # !!! Key operation: Alternating between "within-view" and "cross-view" attention computation !!!
            if i % 2 == 0:
                # Even layers: "within-view" attention (Spatial Attention)
                # Reshape data to (B*N, H*W, C), attention will be computed among patches within each view.
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                # Odd layers: "cross-view" attention (Cross-View Attention)
                # Reshape data to (B, N*H*W, C), attention will be computed among all patches across all views, enabling information exchange.
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            # Save output from the last two layers of decoder for subsequent task heads
            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        # Concatenate outputs from the last two layers as final fused features
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs, input_ids=None, attention_mask=None):
        """
        Model forward pass function.

        Args:
            imgs (torch.Tensor): Input image batch, shape (B, N, 3, H, W).
            input_ids (torch.Tensor, optional): Text input token IDs, shape (B, L).
            attention_mask (torch.Tensor, optional): Text input attention mask, shape (B, L).

        Returns:
            dict: Dictionary containing all prediction results.
        """
        # 1. Image normalization
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # 2. Encoder: Extract features from each view
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # 3. Decoder: Fuse multi-view features
        # `hidden` shape: (B*N, H*W + num_reg, 2*C)
        # `pos` shape: (B*N, H*W + num_reg, 2)
        hidden, pos = self.decode(hidden, N, H, W)

        # 4. Pass through task heads in parallel
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        # Initialize return dictionary
        output = {}

        # 5. Referring segmentation (if enabled)
        if self.use_referring_segmentation:
            # 5.1. Encode text
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state # (B, L, 768)
            text_embeds_proj = self.text_proj(text_embeds) # (B, L, C)

            # 5.2. Multimodal interaction
            # Reshape visual features for batch cross-attention
            # `hidden` from decode is (B*N, num_patches, 2*C), we need the final layer's output
            # Let's use the input to the task heads, which is `hidden`
            hw = (H // self.patch_size) * (W // self.patch_size)
            visual_feats = hidden[:, self.patch_start_idx:, :self.dec_embed_dim] # Use first half of features, (B*N, hw, C)
            visual_feats = visual_feats.reshape(B, N * hw, self.dec_embed_dim)

            # Text features as Query, visual features as Key/Value
            fused_text_feats = self.multimodal_decoder(tgt=text_embeds_proj, memory=visual_feats, tgt_key_padding_mask=(attention_mask == 0))

            # 5.3. Generate segmentation mask
            # Use first token (similar to [CLS]) features as representation of entire description
            seg_token = self.seg_token_head(fused_text_feats[:, 0, :]) # (B, C)

            with torch.amp.autocast(device_type='cuda', enabled=False):
                # Matrix multiplication of segmentation token with visual features of each view to generate low-resolution mask
                # visual_feats for mask: (B, N*hw, C) -> (B, C, N*hw)
                mask_preds_low_res = torch.einsum('bc,bnc->bn', seg_token, visual_feats)
            mask_preds_low_res = mask_preds_low_res.reshape(B, N, patch_h, patch_w)

            # 5.4. Upsample to original resolution
            mask_preds = F.interpolate(
                mask_preds_low_res,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            output['referring_mask_pred'] = mask_preds

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # --- Local point cloud prediction ---
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z) # Ensure depth is positive through exp
            local_points = torch.cat([xy * z, z], dim=-1) # Compute local 3D points from predicted xy and z

            # --- Confidence prediction ---
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # --- Camera pose prediction ---
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # 5. Final processing: Transform local point cloud to global coordinate system using predicted camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        output.update(dict(
            points=points, # Point cloud in global coordinate system
            local_points=local_points, # Local point cloud in camera coordinate system
            conf=conf, # Confidence
            camera_poses=camera_poses, # Predicted camera poses
        ))
        return output
