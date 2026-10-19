from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_ref2va_compat import (
    ComfyH3CompatibilityError,
    ComfyH3Runtime,
    _aligned_target_shape,
)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (5, (5, 2, 8)),
        (120, (124, 37, 207)),
        (124, (124, 37, 207)),
        (125, (141, 42, 235)),
    ],
)
def test_compat_target_shape_matches_native_h3_grid(requested, expected):
    assert _aligned_target_shape(requested) == expected


def test_compat_runtime_builds_nested_av_latent_without_private_node_helpers():
    class NestedTensor:
        def __init__(self, streams):
            self.streams = streams

    runtime = ComfyH3Runtime(
        audio=SimpleNamespace(__name__="comfy.audio", resample=lambda value, *_: value),
        model_management=SimpleNamespace(
            __name__="comfy.model_management", intermediate_device=lambda: "cpu"
        ),
        nested_tensor=SimpleNamespace(
            __name__="comfy.nested_tensor", NestedTensor=NestedTensor
        ),
        utils=SimpleNamespace(__name__="comfy.utils"),
    )
    latent, frame_count = runtime.empty_av_latent(1344, 768, 120)
    video, audio = latent["samples"].streams
    assert frame_count == 124
    assert video.shape == (1, 24, 37, 48, 84)
    assert audio.shape == (1, 32, 2, 207)


def test_compat_runtime_reports_missing_capability_clearly():
    runtime = ComfyH3Runtime(
        audio=SimpleNamespace(__name__="comfy.audio"),
        model_management=SimpleNamespace(__name__="comfy.model_management"),
        nested_tensor=SimpleNamespace(__name__="comfy.nested_tensor"),
        utils=SimpleNamespace(__name__="comfy.utils"),
    )
    with pytest.raises(ComfyH3CompatibilityError, match="NestedTensor"):
        runtime.empty_av_latent(1344, 768, 120)


def test_compat_audio_encoder_validates_native_latent_contract():
    runtime = ComfyH3Runtime(
        audio=SimpleNamespace(
            __name__="comfy.audio", resample=lambda waveform, *_: waveform
        ),
        model_management=SimpleNamespace(__name__="comfy.model_management"),
        nested_tensor=SimpleNamespace(__name__="comfy.nested_tensor"),
        utils=SimpleNamespace(__name__="comfy.utils"),
    )
    vae = SimpleNamespace(
        audio_sample_rate=32000, encode=lambda _waveform: torch.zeros(1, 2, 3)
    )
    with pytest.raises(ComfyH3CompatibilityError, match="latent shape"):
        runtime.encode_ref_audio(
            vae, {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
        )
