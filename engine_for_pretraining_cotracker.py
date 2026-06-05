import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None, cotracker_model=None,
                    ratio_motion=0.15, mask_type="normal", wandb_run=None, feature_extraction_model=None,
                    target_type="pixel"):
    print(f"Masking type: {mask_type}, ratio_motion: {ratio_motion}, target_type: {target_type}")
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()
    imagenet_mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).view(1, -1, 1, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).view(1, -1, 1, 1, 1)

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step
        should_log_scalars = (step % print_freq == 0) or (step == len(data_loader) - 1)
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch

        sampled_mask_type = _sample_mask_type(mask_type)
        pred_tracks = None
        with torch.no_grad():
            unnorm_videos = videos * imagenet_std + imagenet_mean

            if sampled_mask_type == "cotracker_motion_bins":
                denormalized_video_cotracker = unnorm_videos.mul(255.0).float()
                bool_masked_pos, pred_tracks = create_mask_cotracker_motion_bins(
                    denormalized_video_cotracker, cotracker_model, grid_size=14, ratio_motion=ratio_motion, K=data_loader._orig_loader.dataset.transform.masked_position_generator.visible_per_frame
                )
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool).clone()

        # ----------------- Prepare pixel reconstruction labels -----------------
        if target_type == "pixel":
            with torch.no_grad():
                if normlize_target:
                    videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                                            p0=2, p1=patch_size, p2=patch_size)
                    videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                        ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                else:
                    videos_patch = rearrange(unnorm_videos,
                                            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                                            p0=2, p1=patch_size, p2=patch_size)
                B, _, C = videos_patch.shape
                labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        else:
            with torch.no_grad():
                permuted_video = videos.permute(0, 2, 1, 3, 4)
                bs, nf, _, _, _ = permuted_video.shape
                permuted_video = permuted_video[:, ::2].flatten(0, 1)
                features = feature_extraction_model(permuted_video)

                ### ViT-L 14
                if target_type == "clip_vit_l14":
                    b, np, dim = features.shape
                    features = features.reshape(b, 16, 16, dim).permute(0, 3, 1, 2)
                    features = F.interpolate(
                            features,
                            size=(14,14),
                            mode='bilinear')
                    features = features.flatten(2, -1).permute(0, 2, 1)
                ###

                _, np, dim = features.shape
                features = features.reshape(bs, nf//2, np, dim)

                features_squeeze = rearrange(features, 'b n o c -> b (n o) c')
                if normlize_target:
                    labels = (features_squeeze - features_squeeze.mean(dim=-2, keepdim=True)
                        ) / (features_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                else:
                    labels = features_squeeze
                B, _, C = labels.shape
                labels = labels[bool_masked_pos].reshape(B, -1, C)

        # ----------------- CoTracker forward -----------------
        with torch.no_grad():
            if pred_tracks is None:
                denormalized_video_cotracker = unnorm_videos.mul_(255.0).float()
                denormalized_video_cotracker = denormalized_video_cotracker.permute(0, 2, 1, 3, 4)
                denormalized_video_cotracker = denormalized_video_cotracker[:, ::2]

            ### BLOCK FOR UPSAMPLIG ###
            if args.upsample:
                if sampled_mask_type == "normal":
                    pred_tracks, _ = cotracker_model.forward(denormalized_video_cotracker)
                pred_tracks = spatially_interpolate_trajs(pred_tracks, orig_hw=14, upsample_factor=2)
                pred_tracks = pred_tracks[:, 1:, :, :] - pred_tracks[:, :-1, :, :]
                pred_tracks = torch.concat((pred_tracks, pred_tracks[:, -1, :, :].unsqueeze(1)), 1)
                b,t,n,c = pred_tracks.shape
                pred_tracks = pred_tracks.reshape(b,t,28,28,2)
                pred_tracks = rearrange(pred_tracks, 'b t h w c -> b c t h w')
                pred_tracks = rearrange(pred_tracks, 'b c (t t2) (h h2) (w w2) -> b (t h w) (t2 h2 w2) c', t2=1, h2=2, w2=2)
            ### END BLOCK FOR UPSAMPLING ###

            ### NORMAL BLOCK ###
            else:
                if sampled_mask_type == "normal":
                    pred_tracks, _ = cotracker_model.forward(denormalized_video_cotracker)
                pred_tracks = pred_tracks[:, 1:, :, :] - pred_tracks[:, :-1, :, :]
                pred_tracks = torch.concat((pred_tracks, pred_tracks[:, -1, :, :].unsqueeze(1)), 1)

                b, t, n, c = pred_tracks.shape

                pred_tracks = pred_tracks.reshape(b, t, 28, 28, 2)
                pred_tracks = rearrange(pred_tracks, 'b t h w c -> b c t h w')
                pred_tracks = rearrange(pred_tracks, 'b c (t t2) (h h2) (w w2) -> b (t h w) (t2 h2 w2) c',
                                        t2=1, h2=2, w2=2)
            ### END NORMAL BLOCK ###

            if args.cotracker_norm == "patch":
                pred_tracks_norm = (pred_tracks - pred_tracks.mean(dim=-2, keepdim=True)
                    ) / (pred_tracks.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                pred_tracks_patch = rearrange(pred_tracks_norm, 'b n p c -> b n (p c)')
            else:
                pred_tracks_patch = rearrange(pred_tracks, 'b n p c -> b n (p c)')

            B, _, C = pred_tracks_patch.shape
            labels_trajs = pred_tracks_patch[bool_masked_pos].reshape(B, -1, C)

        labels = labels.clone()
        labels_trajs = labels_trajs.clone()

        # ----------------- Model forward -----------------
        with torch.autocast(device_type='cuda'):
            outputs_pixel, outputs_trajs = model(videos, bool_masked_pos)
            loss_1 = loss_func(input=outputs_pixel, target=labels)
            loss_2 = loss_func(input=outputs_trajs, target=labels_trajs)
            if args.traj_only:
                loss = loss_2
            else:
                loss = loss_1 + args.loss_lambda * loss_2

        loss_value = loss.item() if should_log_scalars else None

        if loss_value is not None and not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if loss_value is not None:
            metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if wandb_run is not None and should_log_scalars:
            wandb_run.log({"loss": loss_value, 
                             "loss_1": loss_1.item(),
                             "loss_2": loss_2.item(),
                             "loss_scale": loss_scale_value, 
                             "lr": max_lr, 
                             "min_lr": min_lr, 
                             "weight_decay": weight_decay_value, 
                             "grad_norm": grad_norm,
                             "lambda": args.loss_lambda})

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def spatially_interpolate_trajs(traj, orig_hw=28, upsample_factor=2):
    """
    Interpolates spatial grid of trajectories from N=H*W to N'=H'*W'.
    Args:
        traj: [B, T, N, 2] (trajectory per spatial location)
        orig_hw: original spatial grid size (e.g., 28)
        upsample_factor: scale to increase spatial resolution (2 → double each dim)
    Returns:
        [B, T, N', 2] where N' = (H*upsample)^2
    """
    B, T, N, C = traj.shape
    H, W = orig_hw, orig_hw
    H_new, W_new = H * upsample_factor, W * upsample_factor
    N_new = H_new * W_new
    traj_grid = traj.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
    traj_interp = F.interpolate(traj_grid, size=(T, H_new, W_new), mode='trilinear', align_corners=True)
    traj_interp = traj_interp.permute(0, 2, 3, 4, 1).contiguous().view(B, T, N_new, C)
    return traj_interp

def _resolve_ratio_motion(ratio_motion):
    if isinstance(ratio_motion, (list, tuple)):
        if not ratio_motion:
            raise ValueError("ratio_motion list cannot be empty.")
        lower = float(min(ratio_motion))
        upper = float(max(ratio_motion))
        ratio_motion = lower if lower == upper else random.uniform(lower, upper)
    elif isinstance(ratio_motion, torch.Tensor):
        ratio_motion = float(ratio_motion.item())

    if not isinstance(ratio_motion, (int, float)):
        raise TypeError("ratio_motion must be a float, int, or a list/tuple of numbers.")
    return float(min(1.0, max(0.0, ratio_motion)))

def _sample_mask_type(mask_type):
    if isinstance(mask_type, (list, tuple)):
        if not mask_type:
            raise ValueError("mask_type list cannot be empty.")
        return random.choice(mask_type)
    return mask_type

@torch.no_grad()
def create_mask_cotracker_motion_bins(tracker_frames, cotracker, grid_size=14, ratio_motion=0.5, K=20):
    """
    Split tokens into two equal-sized bins based on motion magnitude (high vs. low),
    then draw K1 samples uniformly from the high-motion bin and K2 samples uniformly
    from the low-motion bin. The selected spatial indices are repeated across all
    time pairs (mask tubing).
    """
    device = tracker_frames.device

    pred_tracks, _ = cotracker(tracker_frames.permute(0, 2, 1, 3, 4)[:, ::2])
    B, T, N, _ = pred_tracks.shape

    source_grid_size = math.isqrt(N)
    if source_grid_size * source_grid_size != N:
        raise ValueError(f"Expected square grid; got N={N} which is not a perfect square")

    target_grid_size = source_grid_size if grid_size is None else grid_size
    if source_grid_size % target_grid_size != 0:
        raise ValueError(
            f"Cannot map CoTracker grid {source_grid_size}x{source_grid_size} "
            f"to {target_grid_size}x{target_grid_size}"
        )
    reduction_factor = source_grid_size // target_grid_size

    coords = pred_tracks[:, :-1, :, :]
    coords_next = pred_tracks[:, 1:, :, :]
    diff = coords_next - coords
    norm = torch.norm(diff, dim=-1)

    mean_disp = torch.mean(norm, dim=1)
    if reduction_factor > 1:
        mean_disp = mean_disp.reshape(
            B,
            target_grid_size,
            reduction_factor,
            target_grid_size,
            reduction_factor,
        ).mean(dim=(2, 4))
    mean_disp = mean_disp.reshape(B, target_grid_size * target_grid_size)
    target_points = target_grid_size * target_grid_size

    if K <= 0 or K > target_points:
        raise ValueError(f"K must be in [1, N={target_points}]")

    ratio_motion = _resolve_ratio_motion(ratio_motion)
    K1 = int(ratio_motion * K)
    K1 = max(0, min(K1, K))
    K2 = K - K1

    sorted_idx = torch.argsort(mean_disp, dim=1, descending=True)
    high_count = max(1, math.ceil(target_points / 2))
    high_count = min(high_count, target_points)
    low_count = target_points - high_count

    high_bin = sorted_idx[:, :high_count]
    low_bin = sorted_idx[:, high_count:]

    def sample_uniform(bin_indices, num_samples, fallback=None):
        bin_size = bin_indices.shape[1]
        if num_samples <= 0:
            return torch.empty(B, 0, dtype=torch.long, device=device)
        if bin_size == 0:
            if fallback is not None and fallback.shape[1] > 0:
                return sample_uniform(fallback, num_samples)
            raise ValueError("Cannot sample from an empty bin")

        take = min(num_samples, bin_size)
        if take > 0:
            rand = torch.rand(B, bin_size, device=device)
            topk = torch.topk(rand, k=take, dim=1, sorted=False).indices
            samples = bin_indices.gather(1, topk)
        else:
            samples = torch.empty(B, 0, dtype=torch.long, device=device)

        if num_samples > take:
            extra = num_samples - take
            repeat_idx = torch.randint(0, bin_size, (B, extra), device=device)
            extra_samples = bin_indices.gather(1, repeat_idx)
            samples = torch.cat([samples, extra_samples], dim=1) if take > 0 else extra_samples

        return samples

    high_samples = sample_uniform(high_bin, K1, fallback=low_bin)
    low_samples = sample_uniform(low_bin, K2, fallback=high_bin)

    final_idx = torch.cat([high_samples, low_samples], dim=1) if K > 0 else torch.empty(B, 0, dtype=torch.long, device=device)

    base_mask = torch.ones(B, target_points, dtype=torch.bool, device=device)
    if K > 0:
        row_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, final_idx.size(1))
        base_mask[row_ids, final_idx] = False

    mask = base_mask.unsqueeze(1).expand(B, T, target_points)

    return mask, pred_tracks
