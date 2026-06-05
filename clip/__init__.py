from pathlib import Path
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WEIGHTS_DIR = _REPO_ROOT / "clip_weights"

_PRETRAINED_PATHS = {
    "CLIP-ViT-B/16": _WEIGHTS_DIR / "ViT-B-16.pt",
    "CLIP-ViT-L/14": _WEIGHTS_DIR / "ViT-L-14.pt",
}


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        drop_path: float = 0.0,
        attn_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("drop1", nn.Dropout(dropout)),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ("drop2", nn.Dropout(dropout)),
                ]
            )
        )
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        drop_path: float = 0.0,
        checkpoint_num: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        drop_path_values = torch.linspace(0, drop_path, layers).tolist()
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    drop_path=drop_path_values[idx],
                    dropout=dropout,
                )
                for idx in range(layers)
            ]
        )
        self.checkpoint_num = checkpoint_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, block in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: Optional[int] = None,
        kernel_size: int = 1,
        num_frames: int = 8,
        drop_path: float = 0.0,
        checkpoint_num: int = 0,
        dropout: float = 0.0,
        temp_embed: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3,
            width,
            (kernel_size, patch_size, patch_size),
            (kernel_size, patch_size, patch_size),
            (0, 0, 0),
            bias=False,
        )

        scale = width ** -0.5
        num_patches = (input_resolution // patch_size) ** 2
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            drop_path=drop_path,
            checkpoint_num=checkpoint_num,
            dropout=dropout,
        )
        self.ln_post = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

    def forward(
        self,
        x: torch.Tensor,
        masking_prob: float = 0.0,
        return_embed: bool = False,
    ) -> torch.Tensor:
        x = self.conv1(x)
        batch_size, channels, num_frames, height, width = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(batch_size * num_frames, height * width, channels)

        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        cls_tokens = x[:batch_size, :1, :]
        x = x[:, 1:]
        x = x.reshape(batch_size, num_frames, height * width, channels)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * height * width, num_frames, channels)

        if hasattr(self, "temporal_positional_embedding"):
            if x.size(1) == 1:
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding

        x = x.reshape(batch_size, height * width, num_frames, channels)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, height * width * num_frames, channels)

        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = self.ln_post(x)

        if return_embed:
            return x.permute(1, 0, 2)[:, 1:]
        return x

    def mask_tokens(self, inputs: torch.Tensor, masking_prob: float = 0.0) -> torch.Tensor:
        batch_size, length, dim = inputs.shape
        num_masked = int(masking_prob * length)
        masked_indices = torch.zeros(batch_size, length, device=inputs.device)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :num_masked]
        batch_indices = torch.arange(batch_size, device=inputs.device).unsqueeze(-1).expand_as(indices)
        masked_indices[batch_indices, indices] = 1
        masked_indices = masked_indices.bool()
        return inputs[~masked_indices].reshape(batch_size, -1, dim)


def inflate_weight(weight_2d: torch.Tensor, time_dim: int, center: bool = True) -> torch.Tensor:
    if center:
        weight_3d = torch.zeros(*weight_2d.shape, device=weight_2d.device, dtype=weight_2d.dtype)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    input_resolution: int = 224,
    patch_size: int = 16,
    center: bool = True,
) -> None:
    model_state_dict = model.state_dict()
    state_dict = dict(state_dict)

    for key, value in list(state_dict.items()):
        if key not in model_state_dict:
            continue
        if value.shape == model_state_dict[key].shape:
            continue
        if len(model_state_dict[key].shape) <= 2:
            continue
        time_dim = model_state_dict[key].shape[2]
        state_dict[key] = inflate_weight(value, time_dim, center=center)

    pos_embed = state_dict["positional_embedding"]
    num_patches = (input_resolution // patch_size) ** 2
    original_size = int((pos_embed.shape[-2] - 1) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if original_size != new_size:
        raise ValueError(
            f"Unsupported positional embedding resize from {original_size} to {new_size}."
        )

    model.load_state_dict(state_dict, strict=False)


def _create_vision_transformer(
    *,
    pretrained: bool | str,
    pretrained_name: str,
    input_resolution: int,
    patch_size: int,
    width: int,
    layers: int,
    heads: int,
    output_dim: int,
    kernel_size: int,
    num_frames: int,
    drop_path: float,
    checkpoint_num: int,
    dropout: float,
    center: bool,
) -> VisionTransformer:
    model = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=output_dim,
        kernel_size=kernel_size,
        num_frames=num_frames,
        drop_path=drop_path,
        checkpoint_num=checkpoint_num,
        dropout=dropout,
    )

    if pretrained:
        model_name = pretrained if isinstance(pretrained, str) else pretrained_name
        checkpoint_path = _PRETRAINED_PATHS[model_name]
        print(f"Loading pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        load_state_dict(
            model,
            state_dict,
            input_resolution=input_resolution,
            patch_size=patch_size,
            center=center,
        )

    return model.eval()


def clip_joint_b16(
    pretrained: bool | str = True,
    input_resolution: int = 224,
    kernel_size: int = 1,
    center: bool = True,
    num_frames: int = 8,
    drop_path: float = 0.0,
    checkpoint_num: int = 0,
    dropout: float = 0.0,
) -> VisionTransformer:
    return _create_vision_transformer(
        pretrained=pretrained,
        pretrained_name="CLIP-ViT-B/16",
        input_resolution=input_resolution,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        kernel_size=kernel_size,
        num_frames=num_frames,
        drop_path=drop_path,
        checkpoint_num=checkpoint_num,
        dropout=dropout,
        center=center,
    )


def clip_joint_l14(
    pretrained: bool | str = True,
    input_resolution: int = 224,
    kernel_size: int = 1,
    center: bool = True,
    num_frames: int = 8,
    drop_path: float = 0.0,
    checkpoint_num: int = 0,
    dropout: float = 0.0,
) -> VisionTransformer:
    return _create_vision_transformer(
        pretrained=pretrained,
        pretrained_name="CLIP-ViT-L/14",
        input_resolution=input_resolution,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        kernel_size=kernel_size,
        num_frames=num_frames,
        drop_path=drop_path,
        checkpoint_num=checkpoint_num,
        dropout=dropout,
        center=center,
    )


CLIP_MODEL_SPECS = {
    "clip_vit_b16": {
        "pretrained": "CLIP-ViT-B/16",
        "feature_dim": 768,
        "builder": clip_joint_b16,
    },
    "clip_vit_l14": {
        "pretrained": "CLIP-ViT-L/14",
        "feature_dim": 1024,
        "builder": clip_joint_l14,
    },
}

CLIP_FEATURE_DIM = CLIP_MODEL_SPECS["clip_vit_b16"]["feature_dim"]


class ClipFeatureExtractor(nn.Module):
    """Minimal wrapper used by TrackMAE pretraining."""

    def __init__(self, model_name: str = "clip_vit_b16"):
        super().__init__()
        if model_name not in CLIP_MODEL_SPECS:
            supported = ", ".join(sorted(CLIP_MODEL_SPECS))
            raise ValueError(f"Unsupported CLIP model '{model_name}'. Supported models: {supported}")

        spec = CLIP_MODEL_SPECS[model_name]
        model = spec["builder"](
            pretrained=spec["pretrained"],
            input_resolution=224,
            kernel_size=1,
            center=True,
            num_frames=1,
            drop_path=0.0,
            checkpoint_num=0,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model
        self.model_name = model_name
        self.feature_dim = spec["feature_dim"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(2)
        elif images.ndim == 5:
            images = images.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(images.shape)}")

        return self.model(images, return_embed=True)


def build_clip_feature_extractor(
    model_name: str = "clip_vit_b16",
) -> ClipFeatureExtractor:
    return ClipFeatureExtractor(model_name=model_name)


def get_clip_feature_dim(model_name: str = "clip_vit_b16") -> int:
    if model_name not in CLIP_MODEL_SPECS:
        supported = ", ".join(sorted(CLIP_MODEL_SPECS))
        raise ValueError(f"Unsupported CLIP model '{model_name}'. Supported models: {supported}")

    return CLIP_MODEL_SPECS[model_name]["feature_dim"]
