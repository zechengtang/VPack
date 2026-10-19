from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from comfyui_ref2va import (
    COMPRESSION_PROFILES,
    CONDITION_FPS,
    NODE_DISPLAY_NAME_MAPPINGS,
    ROPE_DENSE_INDICES_KEY,
    ROPE_DENSE_LATENT_T_KEY,
    ROPE_MODE_KEY,
    ROPE_SCALE_KEY,
    SparkVPack,
    _apply_physical_time_rope,
    _configure_rope_correction,
    _dense_indices_from_source_positions,
    _dense_stride_indices,
    _hybrid_tail_sample_positions,
    _latent_pixel_starts,
    _chunk_phase_indices,
    _compression_dense_indices,
    _chunk_balanced_indices,
    _patch_model_for_physical_time,
    _prepare_reference_image,
    _prepare_reference_video,
    _resample_cfr,
    _resolve_condition_canvas,
    _sample_video_for_qwen,
    _video_rope_span,
    _video_rope_offsets,
)
from comfyui_ref2va_compat import ComfyH3CompatibilityError


class MiniMaxH3Model:
    pass


class _FakeModelPatcher:
    def __init__(self, diffusion_model=None, model_options=None):
        self.diffusion_model = diffusion_model or MiniMaxH3Model()
        self.model_options = model_options or {"transformer_options": {}}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion_model

    def clone(self):
        return _FakeModelPatcher(self.diffusion_model, self.model_options.copy())

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def test_ref2va_480p_canvas_matches_experiment_geometry():
    assert _resolve_condition_canvas(1920, 1080) == (832, 480)
    assert _resolve_condition_canvas(1080, 1920) == (480, 832)
    assert _resolve_condition_canvas(
        1920, 1080, short_edge=768, long_edge_cap=1344
    ) == (1344, 768)


def test_ref2va_cfr_24_to_12_matches_nearest_slot_drop():
    frames = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1).expand(-1, 1, 1, 3)
    result = _resample_cfr(frames, 24.0, CONDITION_FPS)
    assert result[:, 0, 0, 0].tolist() == [0.0, 2.0, 4.0, 6.0]


def test_hybrid_tail_keeps_real_dense_tail_on_valid_grid():
    positions = _hybrid_tail_sample_positions(
        192, source_fps=24.0, condition_fps=12.0
    )
    assert positions == list(range(0, 170, 2)) + list(range(170, 192))
    assert len(positions) == 107
    indices = _dense_indices_from_source_positions(positions, 32, 57)
    assert indices[:8] == [0, 1, 3, 6, 8, 10, 11, 13]
    assert indices[-8:] == [48, 50, 51, 52, 53, 54, 55, 56]
    assert len(indices) == 32


def test_partial_tail_96_frames_keeps_29_latent_starts():
    starts = _latent_pixel_starts(32)
    assert sum(start < 96 for start in starts) == 29
    assert starts[28] == 94
    assert starts[29] == 98
    even_dense_indices = list(range(0, 57, 2))
    assert len(even_dense_indices) == 29
    assert even_dense_indices[-1] == 56


def test_dense_stride_indices_keep_native_endpoint():
    assert _dense_stride_indices(57, 3) == list(range(0, 55, 3)) + [56]
    assert _dense_stride_indices(57, 4) == list(range(0, 57, 4))


def test_chunk_balanced_29_preserves_all_heads_and_spreads_budget():
    indices = _chunk_balanced_indices(57, 29)
    assert len(indices) == 29
    assert set(range(0, 57, 5)).issubset(indices)
    counts = [
        sum(start <= index < min(start + 5, 57) for index in indices)
        for start in range(0, 57, 5)
    ]
    assert counts == [3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2]


def test_ref2va_five_second_condition_floors_to_56_frames():
    frames = torch.zeros(124, 9, 16, 3)
    resize_call = {}

    def resize(value, width, height):
        resize_call.update(width=width, height=height)
        return value

    result = _prepare_reference_video(
        frames,
        source_fps=24.0,
        target_frame_count=124,
        resize=resize,
    )
    assert result.shape[0] == 56
    assert resize_call == {"width": 832, "height": 480}


def test_ref2va_uniform_time_covers_the_complete_duration():
    frames = torch.arange(192, dtype=torch.float32).view(192, 1, 1, 1).expand(-1, 1, 1, 3)
    result = _prepare_reference_video(
        frames,
        source_fps=24.0,
        target_frame_count=192,
        resize=lambda value, _width, _height: value,
        condition_fps=12.0,
        cover_full_duration=True,
    )
    selected = result[:, 0, 0, 0].to(torch.long)
    assert result.shape[0] == 90
    assert selected[0].item() == 0
    assert selected[-1].item() == 190
    assert set(torch.diff(selected).tolist()) == {2, 3}
    assert selected[:8].tolist() == [0, 3, 5, 7, 9, 11, 13, 16]
    qwen_frames, timestamps = _sample_video_for_qwen(result, 90 / 8)
    assert qwen_frames.shape[0] == 16
    assert timestamps == [index / 2.0 for index in range(16)]


def test_ref2va_image_uses_native_max_sizing_and_first_frame_only():
    image = torch.zeros(3, 1672, 940, 4)
    resize_call = {}

    def resize(value, width, height):
        resize_call.update(shape=tuple(value.shape), width=width, height=height)
        return value

    result = _prepare_reference_image(
        image,
        width=736,
        height=544,
        size_mode="max",
        resize=resize,
    )
    assert result.shape == (1, 1672, 940, 3)
    assert resize_call == {
        "shape": (1, 1672, 940, 3),
        "width": 928,
        "height": 1664,
    }


def test_ref2va_configurable_fps_drives_frame_count_and_qwen_clock():
    frames = torch.arange(124, dtype=torch.float32).view(124, 1, 1, 1).expand(-1, 9, 16, 3)
    result = _prepare_reference_video(
        frames,
        source_fps=24.0,
        target_frame_count=124,
        resize=lambda value, _width, _height: value,
        condition_fps=15.0,
    )
    assert result.shape[0] == 73
    conditioned = torch.arange(73, dtype=torch.float32).view(73, 1, 1, 1)
    qwen_frames, timestamps = _sample_video_for_qwen(conditioned, 15.0)
    assert qwen_frames.shape[0] == 10
    assert qwen_frames[:, 0, 0, 0].tolist() == [
        0.0,
        8.0,
        15.0,
        23.0,
        30.0,
        38.0,
        45.0,
        53.0,
        60.0,
        68.0,
    ]
    assert timestamps == [index / 2.0 for index in range(10)]


def test_ref2va_rejects_too_short_12fps_reference():
    frames = torch.zeros(40, 9, 16, 3)
    with pytest.raises(ValueError, match="at least 22"):
        _prepare_reference_video(
            frames,
            source_fps=24.0,
            target_frame_count=124,
            resize=lambda value, _width, _height: value,
        )


def test_physical_time_rope_scales_video_and_shifts_following_rows_once():
    origin = 3.0
    second_frame = origin + 5.0 / 3.0
    old_span = _video_rope_span(2)
    target_origin = origin + old_span
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [origin, 0.0, 0.0],
            [origin, 0.0, 1.0],
            [second_frame, 0.0, 0.0],
            [second_frame, 0.0, 1.0],
            [target_origin, 0.0, 0.0],
            [target_origin, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[
            (0, 3, "text"),
            (3, 7, "ref_img"),
            (7, 8, "audio"),
            (8, 9, "video"),
        ],
        position_ids=positions,
    )
    payload = {
        "layout": layout,
        "refs": [
            {
                "kind": "video",
                "latent_t": 2,
                "latent_h": 2,
                "latent_w": 4,
                "ref_audio_t": 0,
                ROPE_SCALE_KEY: 2.0,
            }
        ],
    }

    assert _apply_physical_time_rope(payload)
    assert layout.position_ids[5, 0].item() == pytest.approx(
        origin + 2.0 * (second_frame - origin)
    )
    delta = old_span
    assert layout.position_ids[7, 0].item() == pytest.approx(target_origin + delta)
    first_result = layout.position_ids.clone()
    assert not _apply_physical_time_rope(payload)
    torch.testing.assert_close(layout.position_ids, first_result)


def test_physical_time_rope_keeps_shared_audio_origin_when_audio_is_longer():
    origin = 2.0
    old_span = _video_rope_span(2)
    audio_span = 20
    positions = torch.zeros(2 + audio_span * 2 + 4 + 2, 3, dtype=torch.float64)
    audio_stop = 2 + audio_span * 2
    video_stop = audio_stop + 4
    positions[audio_stop:video_stop, 0] = torch.tensor(
        [origin, origin, origin + 5.0 / 3.0, origin + 5.0 / 3.0]
    )
    positions[video_stop:, 0] = origin + max(audio_span, old_span)
    layout = SimpleNamespace(
        segments=[
            (0, 2, "text"),
            (2, audio_stop, "ref_audio"),
            (audio_stop, video_stop, "ref_img"),
            (video_stop, video_stop + 1, "audio"),
            (video_stop + 1, video_stop + 2, "video"),
        ],
        position_ids=positions,
    )
    payload = {
        "layout": layout,
        "refs": [
            {
                "kind": "video_audio",
                "latent_t": 2,
                "ref_audio_t": audio_span,
                ROPE_SCALE_KEY: 2.0,
            }
        ],
    }
    target_before = positions[video_stop:, 0].clone()
    assert _apply_physical_time_rope(payload)
    torch.testing.assert_close(layout.position_ids[video_stop:, 0], target_before)


def test_dense_index_rope_uses_native_24fps_grid_instead_of_affine_grid():
    compact_t, dense_t, rows_per_frame = 27, 57, 2
    origin = 3.0
    compact_offsets = _video_rope_offsets(compact_t)
    positions = torch.zeros(compact_t * rows_per_frame + 2, 3, dtype=torch.float64)
    positions[1:-1, 0] = origin + compact_offsets.repeat_interleave(rows_per_frame)
    positions[-1, 0] = origin + _video_rope_span(compact_t)
    layout = SimpleNamespace(
        segments=[
            (0, 1, "text"),
            (1, compact_t * rows_per_frame + 1, "ref_img"),
            (compact_t * rows_per_frame + 1, compact_t * rows_per_frame + 2, "video"),
        ],
        position_ids=positions,
    )
    payload = {
        "layout": layout,
        "refs": [{
            "kind": "video",
            "latent_t": compact_t,
            "ref_audio_t": 0,
            ROPE_SCALE_KEY: 2.0,
            ROPE_MODE_KEY: "dense_index",
            ROPE_DENSE_LATENT_T_KEY: dense_t,
        }],
    }

    assert _apply_physical_time_rope(payload)
    expected_indices = torch.round(torch.linspace(0, dense_t - 1, compact_t)).long()
    expected = _video_rope_offsets(dense_t).index_select(0, expected_indices)
    torch.testing.assert_close(
        layout.position_ids[1:-1, 0],
        origin + expected.repeat_interleave(rows_per_frame),
    )
    assert expected_indices[:8].tolist() == [0, 2, 4, 6, 9, 11, 13, 15]
    assert expected[-1].item() == pytest.approx(313.0 + 1.0 / 3.0)
    assert layout.position_ids[-1, 0].item() == pytest.approx(
        origin + _video_rope_span(dense_t)
    )


@pytest.mark.parametrize(
    ("normal_tokens", "expected_count", "expected_head"),
    [
        (2, 35, [0, 1, 3, 5, 6, 8, 10, 11, 13]),
        (1, 24, [0, 1, 5, 6, 10, 11, 15, 16, 20]),
        (0, 12, [0, 5, 10, 15, 20, 25, 30, 35, 40]),
    ],
)
def test_chunk_phase_indices_preserve_every_chunk_head(
    normal_tokens, expected_count, expected_head
):
    indices = _chunk_phase_indices(57, normal_tokens)
    assert len(indices) == expected_count
    assert indices[:9] == expected_head
    assert set(range(0, 57, 5)).issubset(indices)


def test_chunk_phase_one_normal_can_select_phase_two():
    indices = _chunk_phase_indices(27, 1, single_normal_index=2)
    assert indices == [0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25]


def test_spark_vpack_ratios_use_only_the_fixed_phase_profiles():
    assert COMPRESSION_PROFILES == {
        "0/5": (0, 1, 2, 3),
        "1/5": (0, 1, 2),
        "2/5": (0, 2),
        "3/5": (2,),
    }
    assert _compression_dense_indices(12, "0/5") == list(range(12))
    assert _compression_dense_indices(12, "1/5") == [0, 1, 2, 3, 5, 6, 7, 8, 10, 11]
    assert _compression_dense_indices(12, "2/5") == [0, 1, 3, 5, 6, 8, 10, 11]
    assert _compression_dense_indices(12, "3/5") == [0, 3, 5, 8, 10]
    with pytest.raises(ValueError, match="compression_ratio"):
        _compression_dense_indices(12, "4/5")
    with pytest.raises(ValueError, match="compression_ratio"):
        _compression_dense_indices(12, "5/5")


@pytest.mark.parametrize(
    ("phases", "expected_count"),
    [
        ("0,1", 35),
        ("0,2", 35),
        ("0,3", 35),
        ("1,2", 34),
        ("1,3", 34),
        ("2,3", 34),
    ],
)
def test_chunk_phase_two_normal_explicit_phase_pairs(phases, expected_count):
    indices = _chunk_phase_indices(57, 2, normal_indices=phases)
    assert len(indices) == expected_count
    selected_phases = {int(value) for value in phases.split(",")}
    assert indices[: 1 + len(selected_phases)] == [0] + [
        phase + 1 for phase in sorted(selected_phases)
    ]


@pytest.mark.parametrize(("phase", "expected_count"), [(0, 24), (1, 23), (2, 23), (3, 23)])
def test_chunk_phase_one_normal_all_phase_choices(phase, expected_count):
    indices = _chunk_phase_indices(57, 1, normal_indices=str(phase))
    assert len(indices) == expected_count
    assert indices[:2] == [0, phase + 1]


def test_chunk_phase_zero_head_four_normals():
    indices = _chunk_phase_indices(
        57, 4, normal_indices="0,1,2,3", head_tokens=0
    )
    assert len(indices) == 45
    assert indices[:8] == [1, 2, 3, 4, 6, 7, 8, 9]
    assert set(indices).isdisjoint(range(0, 57, 5))


def test_chunk_phase_rope_restores_selected_native_dense_coordinates():
    dense_t = 57
    dense_indices = _chunk_phase_indices(dense_t, 0)
    compact_t = len(dense_indices)
    origin = 4.0
    positions = torch.zeros(compact_t + 2, 3, dtype=torch.float64)
    positions[1:-1, 0] = origin + _video_rope_offsets(compact_t)
    positions[-1, 0] = origin + _video_rope_span(compact_t)
    layout = SimpleNamespace(
        segments=[
            (0, 1, "text"),
            (1, compact_t + 1, "ref_img"),
            (compact_t + 1, compact_t + 2, "video"),
        ],
        position_ids=positions,
    )
    payload = {
        "layout": layout,
        "refs": [{
            "kind": "video",
            "latent_t": compact_t,
            "ref_audio_t": 0,
            ROPE_SCALE_KEY: 1.0,
            ROPE_MODE_KEY: "chunk_phase",
            ROPE_DENSE_LATENT_T_KEY: dense_t,
            ROPE_DENSE_INDICES_KEY: dense_indices,
        }],
    }

    assert _apply_physical_time_rope(payload)
    expected = _video_rope_offsets(dense_t)[dense_indices]
    torch.testing.assert_close(layout.position_ids[1:-1, 0], origin + expected)
    assert layout.position_ids[-1, 0].item() == pytest.approx(
        origin + _video_rope_span(dense_t)
    )


def test_scale_one_dense_stride_still_applies_explicit_rope():
    dense_t = 57
    dense_indices = _dense_stride_indices(dense_t, 3)
    compact_t = len(dense_indices)
    origin = 2.0
    positions = torch.zeros(compact_t + 2, 3, dtype=torch.float64)
    positions[1:-1, 0] = origin + _video_rope_offsets(compact_t)
    positions[-1, 0] = origin + _video_rope_span(compact_t)
    layout = SimpleNamespace(
        segments=[
            (0, 1, "text"),
            (1, compact_t + 1, "ref_img"),
            (compact_t + 1, compact_t + 2, "video"),
        ],
        position_ids=positions,
    )
    payload = {
        "layout": layout,
        "refs": [{
            "kind": "video",
            "latent_t": compact_t,
            "ref_audio_t": 0,
            ROPE_SCALE_KEY: 1.0,
            ROPE_MODE_KEY: "dense_stride_3",
            ROPE_DENSE_LATENT_T_KEY: dense_t,
            ROPE_DENSE_INDICES_KEY: dense_indices,
        }],
    }
    assert _apply_physical_time_rope(payload)
    torch.testing.assert_close(
        layout.position_ids[1:-1, 0],
        origin + _video_rope_offsets(dense_t)[dense_indices],
    )


def test_physical_time_rope_skips_image_and_only_scales_video_reference():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
            [3.0 + 5.0 / 3.0, 0.0, 0.0],
            [3.0 + 5.0 / 3.0, 0.0, 1.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[
            (0, 1, "text"),
            (1, 3, "ref_img"),
            (3, 7, "ref_img"),
            (7, 8, "video"),
        ],
        position_ids=positions,
    )
    payload = {
        "layout": layout,
        "refs": [
            {"kind": "image", "latent_h": 1, "latent_w": 2},
            {
                "kind": "video",
                "latent_t": 2,
                "latent_h": 1,
                "latent_w": 2,
                "ref_audio_t": 0,
                ROPE_SCALE_KEY: 2.0,
            },
        ],
    }
    image_before = positions[1:3].clone()
    assert _apply_physical_time_rope(payload)
    torch.testing.assert_close(layout.position_ids[1:3], image_before)
    assert layout.position_ids[5, 0].item() == pytest.approx(3.0 + 10.0 / 3.0)


def test_physical_time_rope_rejects_unknown_layout_contract_clearly():
    layout = SimpleNamespace(
        segments=[("text", 1), ("ref_img", 1)],
        position_ids=torch.zeros(2, 3, dtype=torch.float64),
    )
    payload = {
        "layout": layout,
        "refs": [
            {
                "kind": "video",
                "latent_t": 1,
                "ref_audio_t": 0,
                ROPE_SCALE_KEY: 2.0,
            }
        ],
    }
    with pytest.raises(ComfyH3CompatibilityError, match="segments contract"):
        _apply_physical_time_rope(payload)


def test_spark_vpack_node_only_exposes_ratio_control():
    inputs = SparkVPack.INPUT_TYPES()
    assert list(inputs["required"]) == ["model", "positive", "compression_ratio"]
    assert "optional" not in inputs
    ratio_input = inputs["required"]["compression_ratio"]
    assert ratio_input[0] == ["0/5", "1/5", "2/5", "3/5"]
    assert ratio_input[1]["default"] == "2/5"
    assert SparkVPack.RETURN_TYPES == ("MODEL", "CONDITIONING")
    assert SparkVPack.FUNCTION == "pack"
    assert NODE_DISPLAY_NAME_MAPPINGS == {"SparkVPack": "Spark-VPack"}


@pytest.mark.parametrize(
    ("ratio", "expected_latent_t"),
    [("1/5", 30), ("2/5", 23), ("3/5", 15)],
)
def test_spark_vpack_only_drops_reference_video_latent_frames(
    ratio, expected_latent_t
):
    video_latent = torch.arange(37).reshape(1, 1, 37, 1, 1)
    image_ref = {"kind": "image", "latent_h": 20, "latent_w": 30, "latent": torch.zeros(1)}
    video_ref = {
        "kind": "video_audio",
        "latent_t": 37,
        "latent_h": 48,
        "latent_w": 84,
        "latent": video_latent,
        "audio_latent": torch.ones(1),
        "ref_audio_t": 10,
    }
    metadata = {"minimax_refs": [image_ref, video_ref], "other": "unchanged"}
    positive = [[torch.zeros(1), metadata]]
    model = _FakeModelPatcher()

    output_model, output_positive = SparkVPack().pack(model, positive, ratio)

    assert output_model is not model
    assert output_positive is not positive
    assert output_positive[0][0] is positive[0][0]
    assert output_positive[0][1]["other"] == "unchanged"
    packed_image, packed_video = output_positive[0][1]["minimax_refs"]
    assert packed_image is image_ref
    assert packed_video["latent_h"] == 48
    assert packed_video["latent_w"] == 84
    assert packed_video["audio_latent"] is video_ref["audio_latent"]
    assert packed_video["latent_t"] == expected_latent_t
    assert packed_video["latent"].shape == (1, 1, expected_latent_t, 1, 1)
    assert packed_video["latent"].flatten().tolist() == _compression_dense_indices(37, ratio)
    assert packed_video[ROPE_MODE_KEY] == "chunk_phase"
    assert packed_video[ROPE_DENSE_LATENT_T_KEY] == 37
    assert packed_video[ROPE_DENSE_INDICES_KEY] == _compression_dense_indices(37, ratio)
    assert video_ref["latent"] is video_latent
    assert video_ref["latent_t"] == 37
    assert ROPE_DENSE_INDICES_KEY not in video_ref


def test_spark_vpack_requires_encoded_reference_video():
    model = _FakeModelPatcher()
    with pytest.raises(ValueError, match="encoded reference video"):
        SparkVPack().pack(model, [[torch.zeros(1), {"minimax_refs": []}]], "2/5")


def test_spark_vpack_zero_compression_is_a_noop():
    model = _FakeModelPatcher()
    positive = [[torch.zeros(1), {"minimax_refs": []}]]
    assert SparkVPack().pack(model, positive, "0/5") == (model, positive)


def test_ref2va_model_patch_preserves_existing_wrapper_and_patches_payload():
    calls = []

    def previous_wrapper(apply_model, args):
        calls.append("previous")
        return apply_model(args["input"], args["timestep"], **args["c"])

    model = _FakeModelPatcher(model_options={
        "transformer_options": {},
        "model_function_wrapper": previous_wrapper,
    })
    patched = _patch_model_for_physical_time(model)
    assert patched is not model
    wrapper = patched.model_options["model_function_wrapper"]

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[(0, 1, "text"), (1, 2, "ref_img"), (2, 3, "video")],
        position_ids=positions,
    )
    payload = {
        "layout": layout,
        "refs": [{
            "kind": "video",
            "latent_t": 1,
            "ref_audio_t": 0,
            ROPE_SCALE_KEY: 2.0,
        }],
    }

    def apply_model(input_value, timestep, **conditioning):
        assert input_value == "input"
        assert timestep == "sigma"
        assert conditioning["minimax_payload"] is payload
        return "result"

    result = wrapper(
        apply_model,
        {
            "input": "input",
            "timestep": "sigma",
            "c": {"minimax_payload": payload},
            "cond_or_uncond": [0],
        },
    )
    assert result == "result"
    assert calls == ["previous"]
    assert layout._compact_ref2va_physical_time_applied


def test_disable_rope_correction_is_a_clean_opt_in_ablation():
    model = _FakeModelPatcher()
    ablation_model, ablation_metadata = _configure_rope_correction(model, True)
    assert ablation_model is model
    assert ablation_metadata == {}
    assert "model_function_wrapper" not in model.model_options

    corrected_model, corrected_metadata = _configure_rope_correction(model, False)
    assert corrected_model is not model
    assert corrected_metadata == {ROPE_SCALE_KEY: 2.0, ROPE_MODE_KEY: "affine"}
    assert "model_function_wrapper" in corrected_model.model_options


def test_configurable_condition_fps_updates_rope_scale():
    model = _FakeModelPatcher()
    corrected_model, metadata = _configure_rope_correction(model, False, 15.0)
    assert corrected_model is not model
    assert metadata == {
        ROPE_SCALE_KEY: pytest.approx(1.6),
        ROPE_MODE_KEY: "affine",
    }

    native_rate_model, native_rate_metadata = _configure_rope_correction(
        model, False, 24.0
    )
    assert native_rate_model is model
    assert native_rate_metadata == {}

    dense_model, dense_metadata = _configure_rope_correction(
        model, False, 12.0, "dense_index", 57
    )
    assert dense_model is not model
    assert dense_metadata == {
        ROPE_SCALE_KEY: 2.0,
        ROPE_MODE_KEY: "dense_index",
        ROPE_DENSE_LATENT_T_KEY: 57,
    }

    uniform_model, uniform_metadata = _configure_rope_correction(
        model, False, 11.25, "uniform_time", 57
    )
    assert uniform_model is not model
    assert uniform_metadata == {
        ROPE_SCALE_KEY: pytest.approx(24.0 / 11.25),
        ROPE_MODE_KEY: "uniform_time",
    }

    fullspan_model, fullspan_metadata = _configure_rope_correction(
        model, False, 12.0, "dense_index_fullspan", 57
    )
    assert fullspan_model is not model
    assert fullspan_metadata == {
        ROPE_SCALE_KEY: 2.0,
        ROPE_MODE_KEY: "dense_index_fullspan",
        ROPE_DENSE_LATENT_T_KEY: 57,
    }

    chunk_model, chunk_metadata = _configure_rope_correction(
        model, False, 24.0, "chunk_phase", 57, 2
    )
    assert chunk_model is not model
    assert chunk_metadata == {
        ROPE_SCALE_KEY: 1.0,
        ROPE_MODE_KEY: "chunk_phase",
        ROPE_DENSE_LATENT_T_KEY: 57,
        ROPE_DENSE_INDICES_KEY: _chunk_phase_indices(57, 2),
    }

    stride_model, stride_metadata = _configure_rope_correction(
        model, False, 24.0, "dense_stride_3", 57
    )
    assert stride_model is not model
    assert stride_metadata == {
        ROPE_SCALE_KEY: 1.0,
        ROPE_MODE_KEY: "dense_stride_3",
        ROPE_DENSE_LATENT_T_KEY: 57,
        ROPE_DENSE_INDICES_KEY: _dense_stride_indices(57, 3),
    }

    for target_tokens in (29, 32, 34):
        mode = f"chunk_balanced_{target_tokens}"
        balanced_model, balanced_metadata = _configure_rope_correction(
            model, False, 24.0, mode, 57
        )
        assert balanced_model is not model
        assert balanced_metadata == {
            ROPE_SCALE_KEY: 1.0,
            ROPE_MODE_KEY: mode,
            ROPE_DENSE_LATENT_T_KEY: 57,
            ROPE_DENSE_INDICES_KEY: _chunk_balanced_indices(57, target_tokens),
        }
