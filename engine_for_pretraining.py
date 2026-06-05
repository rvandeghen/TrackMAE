import math
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import wandb
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import ToPILImage


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps: int = 0,
                    lr_schedule_values=None, wd_schedule_values=None, wandb_run=None,
                    feature_extraction_model=None, target_type="pixel"):
    """
    Trains the model for one epoch.
    data_loader can be a standard DataLoader or a CUDAPrefetcher.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            videos, bool_masked_pos = batch
        elif isinstance(batch, dict):
            videos, bool_masked_pos = batch['videos'], batch['bool_masked_pos']
        else:
            raise ValueError("Batch format not recognized. Expected (videos, bool_masked_pos) or dict.")

        if not videos.is_cuda:
            videos = videos.to(device, non_blocking=True)
        if not bool_masked_pos.is_cuda:
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if target_type == "pixel":
            with torch.no_grad():
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
                unnorm_videos = videos * std + mean

                if normlize_target:
                    videos_squeeze = rearrange(
                        unnorm_videos,
                        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                        p0=2, p1=patch_size, p2=patch_size,
                    )
                    videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)) / (
                        videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                    )
                    videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                else:
                    videos_patch = rearrange(
                        unnorm_videos,
                        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                        p0=2, p1=patch_size, p2=patch_size,
                    )

                bsz, _, channels = videos_patch.shape
                labels = videos_patch[bool_masked_pos].reshape(bsz, -1, channels)
        else:
            with torch.no_grad():
                permuted_video = videos.permute(0, 2, 1, 3, 4)
                bsz, num_frames, _, _, _ = permuted_video.shape
                permuted_video = permuted_video[:, ::2].flatten(0, 1)
                features = feature_extraction_model(permuted_video)

                if target_type == "clip_vit_l14":
                    batch_frames, num_patches, dim = features.shape
                    features = features.reshape(batch_frames, 16, 16, dim).permute(0, 3, 1, 2)
                    features = F.interpolate(features, size=(14, 14), mode='bilinear')
                    features = features.flatten(2, -1).permute(0, 2, 1)

                _, num_patches, dim = features.shape
                features = features.reshape(bsz, num_frames // 2, num_patches, dim)
                features.requires_grad = False

            with torch.no_grad():
                features_squeeze = rearrange(features, 'b n o c -> b (n o) c')
                if normlize_target:
                    labels = (features_squeeze - features_squeeze.mean(dim=-2, keepdim=True)) / (
                        features_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                    )
                else:
                    labels = features_squeeze
                bsz, _, channels = labels.shape
                labels = labels[bool_masked_pos].reshape(bsz, -1, channels)

        with torch.autocast(device_type='cuda'):
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = 10.0
        max_lr = 0.0
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

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_run is not None:
            wandb_run.log({
                "loss_1": loss_value,
                "loss_scale": loss_scale_value,
                "lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay_value,
                "grad_norm": grad_norm,
            })

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def visualize_reconstruction(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
                             batch_id: int = 0, save_path: str = None, patch_size: int = 16,
                             wandb_run=None, epoch=None):
    """
    Visualizes reconstruction. data_loader can be a standard DataLoader or a CUDAPrefetcher.
    """
    if not utils.is_main_process():
        return

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    patch_size = (patch_size, patch_size)
    model.to(device)
    model.eval()

    if batch_id is None:
        batch_id = 0
    batch = next(islice(data_loader, batch_id, None))

    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        videos, bool_masked_pos = batch
    elif isinstance(batch, dict):
        videos, bool_masked_pos = batch['videos'], batch['bool_masked_pos']
    else:
        raise ValueError("Batch format not recognized. Expected (videos, bool_masked_pos) or dict.")

    if not videos.is_cuda:
        videos = videos.to(device, non_blocking=True)
    if not bool_masked_pos.is_cuda:
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

    total_frames = videos.shape[2]
    videos = videos[batch_id:batch_id + 1, :, :, :, :]
    bool_masked_pos = bool_masked_pos[batch_id:batch_id + 1, :]

    with torch.no_grad():
        outputs = model(videos, bool_masked_pos)

        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
        ori_img = videos * std + mean
        to_pil = ToPILImage()
        ori_imgs = [to_pil(ori_img[0, :, vid, :, :].cpu()) for vid in range(total_frames)]

        img_squeeze = rearrange(
            ori_img,
            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
            p0=2, p1=patch_size[0], p2=patch_size[1],
        )
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
            img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        )
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        for batch_idx in range(outputs.shape[0]):
            img_patch[batch_idx][bool_masked_pos[batch_idx]] = outputs[batch_idx]

        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(
            mask,
            'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ',
            p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14,
        )

        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        rec_img = rec_img + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(
            rec_img,
            'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
            p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14,
        )
        rec_imgs = [to_pil(rec_img[0, :, vid, :, :].cpu().clamp(0, 0.996)) for vid in range(total_frames)]

        for idx, image in enumerate(rec_imgs):
            image.save(f"{save_path}/rec_img{idx}.jpg")

        img_mask = rec_img * mask
        mask_imgs = [to_pil(img_mask[0, :, vid, :, :].cpu()) for vid in range(total_frames)]

        psnr_list = []
        ssim_list = []
        for ori_im, rec_im, _mask_im in zip(ori_imgs, rec_imgs, mask_imgs):
            ori_np = np.array(ori_im)
            rec_np = np.array(rec_im)

            psnr = peak_signal_noise_ratio(ori_np, rec_np, data_range=ori_np.max() - ori_np.min())
            psnr_list.append(psnr)

            ssim = structural_similarity(
                ori_np,
                rec_np,
                multichannel=True,
                data_range=ori_np.max() - ori_np.min(),
                channel_axis=-1,
            )
            ssim_list.append(ssim)

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")

        if wandb_run is not None:
            table = wandb.Table(columns=["Frame_idx", "Ori_img", "Rec_img", "Mask_img", "PSNR", "SSIM"])
            for idx, (ori_im, rec_im, mask_im, psnr, ssim) in enumerate(
                zip(ori_imgs, rec_imgs, mask_imgs, psnr_list, ssim_list)
            ):
                wandb_ori_img = wandb.Image(ori_im)
                wandb_rec_img = wandb.Image(rec_im)
                wandb_mask_img = wandb.Image(mask_im)
                table.add_data(idx, wandb_ori_img, wandb_rec_img, wandb_mask_img, psnr, ssim)

            wandb_run.log({f"epoch_{epoch}_reconstruction_video": table})
