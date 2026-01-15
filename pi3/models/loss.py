import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from pi3.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

def dice_loss_global(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    eps=1e-6,
    mode: str = 'perview',
    suppress_no_target: bool = True,
    add_global_loss_weight: float = 0.0,
    anneal_global_loss: bool = False,
    current_epoch: int = None,
    total_epochs: int = None,
    only_target=False,
):
    """
    Compute the DICE loss with various strategies.
    Args:
        inputs: A float tensor of shape (B, V, H, W).
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        mode: 'global' or 'perview'. The primary loss calculation method.
        suppress_no_target: If True and mode is 'perview', down-weights views with no target mask.
        add_global_loss_weight: If > 0, adds a global dice loss component to the per-view loss.
        anneal_global_loss: If True, adds a global loss component with weight annealed from 0 to 1 over epochs.
    """
    if mode == 'global':
        return dice_loss_global(inputs, targets, num_masks, eps).mean()
    
    if mode == 'perview':
        inputs_sig = inputs.sigmoid()
        
        # Per-view loss calculation
        inputs_flat = inputs_sig.flatten(2)  # (B, V, H*W)
        targets_flat = targets.flatten(2)  # (B, V, H*W)
        numerator = 2 * (inputs_flat * targets_flat).sum(-1)  # (B, V)
        denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)  # (B, V)
        per_view_loss = 1 - (numerator + eps) / (denominator + eps)  # (B, V)

        if suppress_no_target:
            weights = torch.ones_like(per_view_loss)
            no_target_mask = (targets_flat.sum(-1) == 0)
            
            if no_target_mask.any():
                num_no_target_per_item = no_target_mask.sum(dim=1, keepdim=True)
                
                if only_target:
                    weights_for_no_target = torch.zeros_like(num_no_target_per_item)
                else:
                    weights_for_no_target = 1.0 / torch.clamp(num_no_target_per_item, min=1)
                
                weights = torch.where(no_target_mask, weights_for_no_target, weights)
            
            per_view_loss = per_view_loss * weights
        
        final_loss = per_view_loss.mean()

        if add_global_loss_weight > 0:
            global_loss = dice_loss_global(inputs, targets, num_masks, eps).mean()
            final_loss += add_global_loss_weight * global_loss
        elif anneal_global_loss:
            if current_epoch is not None and total_epochs is not None and total_epochs > 0:
                anneal_weight = current_epoch / total_epochs
                global_loss = dice_loss_global(inputs, targets, num_masks, eps).mean()
                final_loss += anneal_weight * global_loss 
        
        return final_loss

    raise ValueError(f"Unknown dice_loss mode: {mode}")


def iou_score(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
    only_target=True,
):
    """
    Compute the IoU score
    Args:
        inputs: A float tensor of shape (B, V, H, W).
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # binary
    inputs = (inputs > 0.5).float() # (B, V, H, W)
    # flatten H, W
    inputs = inputs.flatten(2)  # (B, V, H*W)
    targets = targets.flatten(2)  # (B, V, H*W)
    numerator = (inputs * targets).sum(-1)  # (B, V)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator  # (B, V)
    score = (numerator + eps) / (denominator + eps)  # (B, V)

    if only_target:
        # Only keep views with targets
        score = score * (targets.sum(-1) > 0).float() # (B, V)
        # Compute IoU score for views with targets, note that some samples may have no targets
        # score = score.sum(-1) / (targets.sum(-1) > 0).float().sum(-1) # (B)
        # If all views have no targets, score is 0; otherwise compute IoU score directly
        score = torch.where((targets.sum(-1) > 0).float().sum(-1) > 0, score.sum(-1) / (targets.sum(-1) > 0).float().sum(-1), torch.zeros_like(score[:, 0])) # (B)

    
    return score.mean()


def iou_score_global(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Compute the IoU score
    Args:
        inputs: A float tensor of arbitrary shape..
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() # (B, V, H, W)
    # binary
    inputs = (inputs > 0.5).float() # (B, V, H, W)
    # flatten
    inputs = inputs.flatten(1) # (B, V*H*W)
    targets = targets.flatten(1) # (B, V*H*W)
    numerator = (inputs * targets).sum(-1)  # (B)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator  # (B)
    score = (numerator + eps) / (denominator + eps)  # (B)
    return score.mean()


def iou_score_global_per_sample(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Computes global IoU for each sample in the batch.
    Returns a tensor of shape (B,).
    """
    inputs = inputs.sigmoid()
    inputs = (inputs > 0.5).float()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator
    score = (numerator + eps) / (denominator + eps)
    return score

def iou_score_per_view(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Computes IoU for each view for each sample in the batch.
    Returns a tensor of shape (B, V).
    """
    inputs = inputs.sigmoid()
    inputs = (inputs > 0.5).float()  # (B, V, H, W)

    # flatten H, W
    inputs_flat = inputs.flatten(2)  # (B, V, H*W)
    targets_flat = targets.flatten(2)  # (B, V, H*W)

    numerator = (inputs_flat * targets_flat).sum(-1)  # (B, V)
    denominator = inputs_flat.sum(-1) + targets_flat.sum(-1) - numerator  # (B, V)

    score_per_view = (numerator + eps) / (denominator + eps)  # (B, V)
    return score_per_view


def iou_score_per_view_avg(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Computes IoU for each view, then averages over views for each sample in the batch.
    Only views with a ground truth mask are considered in the average.
    Returns a tensor of shape (B,).
    """
    inputs = inputs.sigmoid()
    inputs = (inputs > 0.5).float()  # (B, V, H, W)

    # flatten H, W
    inputs_flat = inputs.flatten(2)  # (B, V, H*W)
    targets_flat = targets.flatten(2)  # (B, V, H*W)

    numerator = (inputs_flat * targets_flat).sum(-1)  # (B, V)
    denominator = inputs_flat.sum(-1) + targets_flat.sum(-1) - numerator  # (B, V)

    score_per_view = (numerator + eps) / (denominator + eps)  # (B, V)

    # We should only average over views that have a target mask.
    has_target = targets_flat.sum(-1) > 0  # (B, V)

    # Sum scores only for views with targets
    sum_scores_per_sample = (score_per_view * has_target.float()).sum(dim=-1)  # (B,)

    # Count number of views with targets for each sample
    num_views_with_target = has_target.sum(dim=-1)  # (B,)

    # Average score per sample. Use clamp to avoid division by zero.
    # If a sample has 0 views with target, score is 0.
    avg_score_per_sample = sum_scores_per_sample / torch.clamp(num_views_with_target.float(), min=1.0)  # (B,)

    return avg_score_per_sample


class ReferringMaskLoss(nn.Module):
    def __init__(self, weight_dict=None, layer_weight=0.5, 
                 dice_loss_mode='perview', 
                 perview_suppress_no_target=True, 
                 add_global_loss_weight=0.0, 
                 anneal_global_loss=False, 
                 only_target=False):
        super().__init__()
        self.weight_dict = weight_dict if weight_dict is not None else {'loss_mask': 1, 'loss_dice': 1}
        self.layer_weight = layer_weight
        self.dice_loss_mode = dice_loss_mode
        self.perview_suppress_no_target = perview_suppress_no_target
        self.add_global_loss_weight = add_global_loss_weight
        self.anneal_global_loss = anneal_global_loss
        self.only_target = only_target

    def forward(self, pred, gt, current_epoch=None, total_epochs=None):
        pred_masks = pred['referring_mask_pred']
        gt_masks = gt['referring_masks'] # (B, V, H, W)
        
        num_masks = gt_masks.shape[0] * gt_masks.shape[1]

        losses = {}
        # Compute loss for the final layer
        if self.only_target: # Only supervise views with targets
            bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float(), reduction='none')
            has_target = (gt_masks.sum(dim=(-1, -2), keepdim=True) > 0).float() # (B, V, 1, 1)
            
            bce_loss = (bce_loss * has_target).sum() / (has_target.sum() * pred_masks.shape[-1] * pred_masks.shape[-2] + 1e-8)
            losses["loss_mask"] = bce_loss
        else:
            losses["loss_mask"] = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())
        losses["loss_dice"] = dice_loss(
            pred_masks, gt_masks, num_masks, 
            mode=self.dice_loss_mode, 
            suppress_no_target=self.perview_suppress_no_target,
            add_global_loss_weight=self.add_global_loss_weight,
            anneal_global_loss=self.anneal_global_loss,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            only_target=self.only_target
        )
        losses["iou_score"] = iou_score_global(pred_masks, gt_masks)
        # Calculate per-sample IoUs but do not add them to the main `losses` dict to avoid issues with MetricLogger in training loop.
        iou_global_per_sample = iou_score_global_per_sample(pred_masks, gt_masks)
        iou_per_view_avg = iou_score_per_view_avg(pred_masks, gt_masks)
        iou_per_view = iou_score_per_view(pred_masks, gt_masks)
        # iou score just in frame with target
        losses["iou_score_in_frame_with_target"] = iou_score(pred_masks, gt_masks)
        total_loss = self.weight_dict['loss_mask'] * losses['loss_mask'] + self.weight_dict['loss_dice'] * losses['loss_dice']
        # Record the ratio of samples with no targets, i.e., samples where all views have no targets
        view_has_target = (gt_masks.sum(dim=(-1, -2)) > 0).float() # (B, V)
        sample_no_target = (view_has_target.sum(-1) == 0).float() # (B)
        losses["rate_no_target"] = sample_no_target.mean()

        # Record the ratio of frames with targets / total frames for each sample, then average
        view_has_target = gt_masks.sum(dim=(-1, -2)) > 0 # (B, V)
        rate_view_has_target = view_has_target.float().mean(-1) # (B)
        losses["rate_frame_with_target"] = rate_view_has_target.mean()

        # Record the pixel ratio of targets in each sample
        pixel_rate_per_view = gt_masks.float().mean(dim=(-1, -2)) # (B, V)
        losses["rate_pixel_with_target"] = pixel_rate_per_view.mean()

        # Record the average pixel ratio of targets in frames with targets
        if view_has_target.any():
            rate_pixel_in_target_frame = pixel_rate_per_view[view_has_target].mean()
        else:
            rate_pixel_in_target_frame = torch.tensor(0.0, device=gt_masks.device)
        losses["rate_pixel_in_target_frame"] = rate_pixel_in_target_frame
        # If there are intermediate layer predictions, compute intermediate layer losses
        if 'layer_referring_mask_preds' in pred:
            layer_preds = pred['layer_referring_mask_preds']
            layer_losses = {}
            
            # # Compute loss for each layer
            # intermediate_loss = 0.0
            # if len(layer_preds) > 0:
            #     for i, layer_pred in enumerate(layer_preds):
            #         if self.only_target:
            #             bce_loss = F.binary_cross_entropy_with_logits(layer_pred, gt_masks.float(), reduction='none')
            #             has_target = (gt_masks.sum(dim=(-1, -2), keepdim=True) > 0).float()
            #             bce_loss = (bce_loss * has_target).sum() / (has_target.sum() * layer_pred.shape[-1] * layer_pred.shape[-2] + 1e-8)
            #             layer_losses[f"loss_mask_layer_{i}"] = bce_loss
            #         else:
            #             layer_losses[f"loss_mask_layer_{i}"] = F.binary_cross_entropy_with_logits(layer_pred, gt_masks.float())
            #         layer_losses[f"loss_dice_layer_{i}"] = dice_loss(layer_pred, gt_masks, num_masks, use_global_loss=self.use_global_loss, only_target=self.only_target)
            #         layer_losses[f"iou_score_layer_{i}"] = iou_score_global(layer_pred, gt_masks)

            #         # Accumulate loss for each intermediate layer
            #         intermediate_loss += (
            #             self.weight_dict['loss_mask'] * layer_losses[f"loss_mask_layer_{i}"] + 
            #             self.weight_dict['loss_dice'] * layer_losses[f"loss_dice_layer_{i}"]
            #         )
                
            #     # Compute average loss and add to total loss
            #     total_loss += intermediate_loss / len(layer_preds)
            # Compute loss for each layer
            for i, layer_pred in enumerate(layer_preds):
                if self.only_target:
                    bce_loss = F.binary_cross_entropy_with_logits(layer_pred, gt_masks.float(), reduction='none')
                    has_target = (gt_masks.sum(dim=(-1, -2), keepdim=True) > 0).float()
                    bce_loss = (bce_loss * has_target).sum() / (has_target.sum() * layer_pred.shape[-1] * layer_pred.shape[-2] + 1e-8)
                    layer_losses[f"loss_mask_layer_{i}"] = bce_loss
                else:
                    layer_losses[f"loss_mask_layer_{i}"] = F.binary_cross_entropy_with_logits(layer_pred, gt_masks.float())
                layer_losses[f"loss_dice_layer_{i}"] = dice_loss(
                    layer_pred, gt_masks, num_masks, 
                    mode=self.dice_loss_mode, 
                    suppress_no_target=self.perview_suppress_no_target,
                    add_global_loss_weight=self.add_global_loss_weight,
                    anneal_global_loss=self.anneal_global_loss,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                    only_target=self.only_target
                )
                layer_losses[f"iou_score_layer_{i}"] = iou_score_global(layer_pred, gt_masks)

                # Add intermediate layer loss to total loss, multiplied by weight coefficient
                total_loss += self.layer_weight * (
                    self.weight_dict['loss_mask'] * layer_losses[f"loss_mask_layer_{i}"] + 
                    self.weight_dict['loss_dice'] * layer_losses[f"loss_dice_layer_{i}"]
                )
            
            # Add intermediate layer losses to details
            losses.update(layer_losses)
        
        details = {f"refer_{k}": v.detach() for k, v in losses.items()}
        details['refer_iou_score_per_sample'] = iou_per_view_avg.detach()
        details['refer_iou_score_global_per_sample'] = iou_global_per_sample.detach()
        details['refer_iou_per_view'] = iou_per_view.detach()
        return total_loss, details

class Pi3Loss(nn.Module):
    def __init__(
        self,
        train_conf=False,
        use_referring_segmentation=False,
        referring_loss_weight_dict=None,
        referring_layer_weight=0.5,
        dice_loss_mode='perview',
        perview_suppress_no_target=True,
        add_global_loss_weight=0.0,
        anneal_global_loss=False,
        only_target=False,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
        self.use_referring_segmentation = use_referring_segmentation
        if self.use_referring_segmentation:
            self.referring_mask_loss = ReferringMaskLoss(
                weight_dict=referring_loss_weight_dict,
                layer_weight=referring_layer_weight,
                dice_loss_mode=dice_loss_mode,
                perview_suppress_no_target=perview_suppress_no_target,
                add_global_loss_weight=add_global_loss_weight,
                anneal_global_loss=anneal_global_loss,
                only_target=only_target
            )

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        if self.use_referring_segmentation and gt[0]['referring_mask'] is not None:
            referring_masks = torch.stack([view['referring_mask'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = gt[0]['dataset']

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            referring_masks=referring_masks if self.use_referring_segmentation else None,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw, current_epoch=None, total_epochs=None):
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss if not self.use_referring_segmentation else 0.0
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1 if not self.use_referring_segmentation else 0.0
        details.update(camera_loss_details)

        if self.use_referring_segmentation and 'referring_mask_pred' in pred and 'referring_masks' in gt:
            referring_loss, referring_loss_details = self.referring_mask_loss(pred, gt, current_epoch=current_epoch, total_epochs=total_epochs)
            final_loss += referring_loss
            details.update(referring_loss_details)

        return final_loss, details
