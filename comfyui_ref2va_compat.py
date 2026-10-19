"""Capability-based ComfyUI compatibility adapter for Ref2VA conditioning."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

import torch

TARGET_FPS = 24.0
AUDIO_LATENT_FPS = 40.0


class ComfyH3CompatibilityError(RuntimeError):
    """Raised when the installed ComfyUI lacks a required native H3 capability."""


def _require(module: ModuleType, name: str):
    value = getattr(module, name, None)
    if value is None:
        raise ComfyH3CompatibilityError(
            f"installed ComfyUI is missing {module.__name__}.{name}; "
            "update to a release with native MiniMax-H3 support"
        )
    return value


def _aligned_target_shape(length: int) -> tuple[int, int, int]:
    """Return native H3 target ``(frames, video_t, audio_t)`` without private helpers."""
    frame_count = max(5, int(length))
    frame_count += (5 - frame_count) % 17
    video_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_t = round(frame_count / TARGET_FPS * AUDIO_LATENT_FPS)
    return frame_count, video_t, audio_t


@dataclass(frozen=True)
class ComfyH3Runtime:
    """Small stable surface built from capability-checked ComfyUI modules."""

    audio: ModuleType
    model_management: ModuleType
    nested_tensor: ModuleType
    utils: ModuleType

    def empty_av_latent(
        self, width: int, height: int, length: int, batch_size: int = 1
    ) -> tuple[dict, int]:
        nested_tensor_type = _require(self.nested_tensor, "NestedTensor")
        intermediate_device = _require(self.model_management, "intermediate_device")
        frame_count, video_t, audio_t = _aligned_target_shape(length)
        video = torch.zeros(
            [batch_size, 24, video_t, height // 16, width // 16],
            device=intermediate_device(),
        )
        audio = torch.zeros(
            [batch_size, 32, 2, audio_t],
            device=intermediate_device(),
        )
        return {"samples": nested_tensor_type((video, audio))}, frame_count

    def resize(
        self, image: torch.Tensor, width: int, height: int
    ) -> torch.Tensor:
        common_upscale = _require(self.utils, "common_upscale")
        samples = image[..., :3].movedim(-1, 1)
        samples = common_upscale(samples, width, height, "lanczos", "disabled")
        return samples.movedim(1, -1)

    def encode_ref_audio(self, audio_vae, audio: dict) -> tuple[torch.Tensor, int]:
        resample = _require(self.audio, "resample")
        try:
            waveform = audio["waveform"]
            sample_rate = int(audio["sample_rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "reference audio must contain waveform and sample_rate"
            ) from error
        vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sample_rate != vae_sample_rate:
            waveform = resample(waveform, sample_rate, vae_sample_rate)
        latent = audio_vae.encode(waveform[:1].movedim(1, -1))
        if not torch.is_tensor(latent) or latent.ndim != 4:
            raise ComfyH3CompatibilityError(
                "installed MiniMax-H3 audio VAE returned an unsupported latent shape"
            )
        return latent, int(latent.shape[-1])


def load_comfy_h3_runtime() -> ComfyH3Runtime:
    """Load only capabilities shared by native-H3 ComfyUI versions."""
    modules = {}
    for name in (
        "comfy.audio",
        "comfy.model_management",
        "comfy.nested_tensor",
        "comfy.utils",
    ):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError as error:
            raise ComfyH3CompatibilityError(
                f"installed ComfyUI does not provide {name}; "
                "native MiniMax-H3 support is required"
            ) from error
    return ComfyH3Runtime(
        audio=modules["comfy.audio"],
        model_management=modules["comfy.model_management"],
        nested_tensor=modules["comfy.nested_tensor"],
        utils=modules["comfy.utils"],
    )


__all__ = [
    "ComfyH3CompatibilityError",
    "ComfyH3Runtime",
    "load_comfy_h3_runtime",
]
