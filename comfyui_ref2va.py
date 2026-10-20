"""Pack native MiniMax-H3 Ref2VA reference latents on their temporal axis.

The public node changes no spatial dimensions or generated latents. Its model
wrapper restores retained reference frames' original temporal RoPE coordinates.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch

try:
    from .comfyui_ref2va_compat import (
        ComfyH3CompatibilityError,
        load_comfy_h3_runtime,
    )
except ImportError:  # standalone source-tree tests
    from comfyui_ref2va_compat import (
        ComfyH3CompatibilityError,
        load_comfy_h3_runtime,
    )

log = logging.getLogger(__name__)

TARGET_FPS = 24.0
CONDITION_FPS = 12.0
QWEN_VIDEO_FPS = 2.0
CANVAS_MULTIPLE = 32
CONDITION_SHORT_EDGE = 480
CONDITION_LONG_EDGE_CAP = 832
REF_IMAGE_SHORT_EDGE = 2048
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
COMPRESSION_PROFILES = {
    "0/5": (0, 1, 2, 3),
    "1/5": (0, 1, 2),
    "2/5": (0, 2),
    "3/5": (2,),
}
ROPE_SCALE_KEY = "compact_ref2va_rope_time_scale"
ROPE_MODE_KEY = "compact_ref2va_rope_time_mode"
ROPE_DENSE_LATENT_T_KEY = "compact_ref2va_dense_latent_t"
ROPE_DENSE_INDICES_KEY = "compact_ref2va_dense_indices"
_LAYOUT_PATCH_MARKER = "_compact_ref2va_physical_time_applied"


def _resolve_condition_canvas(
    width: int,
    height: int,
    *,
    short_edge: int = CONDITION_SHORT_EDGE,
    long_edge_cap: int = CONDITION_LONG_EDGE_CAP,
) -> tuple[int, int]:
    """Return a configurable Ref2VA canvas as ``(width, height)``."""
    if width <= 0 or height <= 0:
        raise ValueError("reference-video width and height must be positive")
    if short_edge <= 0 or long_edge_cap < short_edge:
        raise ValueError(
            "condition_short_edge must be positive and no larger than "
            "condition_long_edge_cap"
        )
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = short_edge * ratio, short_edge
    else:
        nominal_width, nominal_height = short_edge, short_edge / ratio
    max_pixels = short_edge * long_edge_cap
    if nominal_width * nominal_height > max_pixels:
        scale = math.sqrt(max_pixels / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    resolved_width = max(
        CANVAS_MULTIPLE,
        round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    resolved_height = max(
        CANVAS_MULTIPLE,
        round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return resolved_width, resolved_height


def _resample_cfr(frames: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    """Nearest-slot CFR conversion matching MiniMax-H3's Diffusers setup step."""
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(
            "ref_video must be an IMAGE batch shaped [frames, height, width, channels]"
        )
    if frames.shape[0] == 0:
        raise ValueError("ref_video contains no frames")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if source_fps == target_fps:
        return frames

    scale = target_fps / source_fps
    frame_ids = torch.arange(frames.shape[0], dtype=torch.float64)
    slots = torch.floor(frame_ids * scale + 0.5).to(torch.long)
    end = math.floor(frames.shape[0] * scale + 0.5)
    counts = torch.diff(
        torch.cat((slots, torch.tensor([end], dtype=torch.long)))
    )
    if bool((counts < 0).any()):
        raise RuntimeError("CFR resampling produced a non-monotonic frame map")
    return torch.repeat_interleave(frames, counts.to(frames.device), dim=0)


def _valid_reference_frame_count(available: int) -> int:
    """Floor to the video VAE's 17*n+5 grid, requiring a temporal clip."""
    if available < 22:
        raise ValueError(
            "the reference must contain at least 22 usable condition frames"
        )
    return ((available - 5) // 17) * 17 + 5


def _valid_reference_frame_count_ceiling(available: int) -> int:
    """Ceil to the video VAE's ``17*n+5`` grid."""
    if available < 5:
        return 5
    return math.ceil((available - 5) / 17) * 17 + 5


def _latent_pixel_starts(latent_frames: int) -> list[int]:
    """Pixel-frame start represented by each H3 temporal latent token."""
    starts: list[int] = []
    position = 0
    for index in range(latent_frames):
        starts.append(position)
        position += FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
    return starts


def _hybrid_tail_sample_positions(
    source_count: int,
    *,
    source_fps: float,
    condition_fps: float,
) -> list[int]:
    """Half-rate prefix plus a dense tail, ending on a legal H3 VAE grid."""
    ratio = source_fps / condition_fps
    if not math.isclose(ratio, 2.0):
        raise ValueError("hybrid_tail_dense currently requires source_fps / condition_fps == 2")
    requested = round(source_count / ratio)
    target = _valid_reference_frame_count_ceiling(requested)
    dense_tail = 2 * (target - requested)
    split = source_count - dense_tail
    if split < 0 or split % 2:
        raise ValueError("hybrid_tail_dense requires an even half-rate prefix")
    positions = list(range(0, split, 2)) + list(range(split, source_count))
    if len(positions) != target:
        raise RuntimeError("hybrid-tail sampling did not reach the expected VAE grid")
    return positions


def _dense_indices_from_source_positions(
    source_positions: list[int],
    compact_latent_frames: int,
    dense_latent_frames: int,
) -> list[int]:
    """Snap compact token starts to their closest native dense token starts."""
    compact_starts = _latent_pixel_starts(compact_latent_frames)
    dense_starts = _latent_pixel_starts(dense_latent_frames)
    if not source_positions or compact_starts[-1] >= len(source_positions):
        raise ValueError("source positions do not cover the compact latent timeline")
    result = []
    for compact_start in compact_starts:
        source_start = source_positions[compact_start]
        result.append(
            min(range(len(dense_starts)), key=lambda index: abs(dense_starts[index] - source_start))
        )
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("source-derived dense indices must be strictly increasing")
    return result


def _prepare_reference_video(
    frames: torch.Tensor,
    *,
    source_fps: float,
    target_frame_count: int,
    resize: Callable[[torch.Tensor, int, int], torch.Tensor],
    condition_fps: float = CONDITION_FPS,
    condition_short_edge: int = CONDITION_SHORT_EDGE,
    condition_long_edge_cap: int = CONDITION_LONG_EDGE_CAP,
    cover_full_duration: bool = False,
) -> torch.Tensor:
    """Resample, duration-limit, VAE-align and resize a reference video."""
    if condition_fps <= 0:
        raise ValueError("condition_fps must be positive")
    requested = round(target_frame_count * condition_fps / TARGET_FPS)
    if cover_full_duration:
        # The VAE only accepts 17*n+5 frames.  Instead of taking a nominal CFR
        # conversion and discarding its tail to reach that grid, distribute the
        # valid frame count over the complete requested time interval.  Keep the
        # opening frame exact; the final sample represents the final time bin.
        source_count = min(
            int(frames.shape[0]),
            max(1, math.ceil(target_frame_count * source_fps / TARGET_FPS)),
        )
        available_at_condition_rate = min(
            requested,
            round(source_count * condition_fps / source_fps),
        )
        valid_frame_count = _valid_reference_frame_count(
            available_at_condition_rate
        )
        sample_positions = torch.floor(
            (torch.arange(valid_frame_count, dtype=torch.float64) + 0.5)
            * source_count
            / valid_frame_count
        ).to(torch.long)
        sample_positions[0] = 0
        sample_positions.clamp_max_(source_count - 1)
        normalized = frames[..., :3].index_select(
            0, sample_positions.to(frames.device)
        )
    else:
        normalized = _resample_cfr(frames[..., :3], source_fps, condition_fps)
        normalized = normalized[:requested]
        try:
            valid_frame_count = _valid_reference_frame_count(normalized.shape[0])
        except ValueError as error:
            minimum_duration = 22 / condition_fps
            raise ValueError(
                f"the {condition_fps:g} fps reference must contain at least 22 "
                f"usable frames (about {minimum_duration:.2f} seconds)"
            ) from error
        normalized = normalized[:valid_frame_count]
    height, width = normalized.shape[1:3]
    canvas_width, canvas_height = _resolve_condition_canvas(
        width,
        height,
        short_edge=condition_short_edge,
        long_edge_cap=condition_long_edge_cap,
    )
    return resize(normalized, canvas_width, canvas_height)


def _prepare_reference_image(
    image: torch.Tensor,
    *,
    width: int,
    height: int,
    size_mode: str,
    resize: Callable[[torch.Tensor, int, int], torch.Tensor],
) -> torch.Tensor:
    """Apply native H3 reference-image sizing without video compression."""
    if image.ndim != 4 or image.shape[0] == 0 or image.shape[-1] < 3:
        raise ValueError(
            "ref_image must be an IMAGE batch shaped [batch, height, width, channels]"
        )
    source_height, source_width = image.shape[1:3]
    if size_mode == "match":
        scale = min(
            1.0,
            math.sqrt((int(width) * int(height)) / (source_width * source_height)),
        )
    elif size_mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(source_width, source_height))
    else:
        raise ValueError("ref_image_size must be either 'match' or 'max'")
    target_width = max(
        CANVAS_MULTIPLE,
        round(source_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    target_height = max(
        CANVAS_MULTIPLE,
        round(source_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return resize(image[:1, ..., :3], target_width, target_height)


def _sample_video_for_qwen(
    frames: torch.Tensor,
    source_fps: float,
    target_fps: float = QWEN_VIDEO_FPS,
) -> tuple[torch.Tensor, list[float]]:
    """Sample frames from time zero onto Qwen's fixed visual clock."""
    if frames.shape[0] == 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Qwen video sampling requires frames and positive frame rates")
    sample_count = max(1, math.ceil(frames.shape[0] * target_fps / source_fps))
    indices = torch.floor(
        torch.arange(sample_count, dtype=torch.float64) * source_fps / target_fps
        + 0.5
    ).to(torch.long)
    indices = indices.clamp_max(frames.shape[0] - 1).to(frames.device)
    timestamps = [index / target_fps for index in range(sample_count)]
    return frames.index_select(0, indices), timestamps


def _video_rope_span(latent_frames: int) -> float:
    return sum(
        FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
        for index in range(latent_frames)
    )


def _video_latent_frame_count(frame_count: int) -> int:
    """Match native MiniMax-H3's 17*n+5 pixel-frame to 5*n+2 latent grid."""
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError("video frame count must be on MiniMax-H3's 17*n+5 grid")
    return 2 if frame_count == 5 else ((frame_count - 5) // 17) * 5 + 2


def _video_rope_offsets(latent_frames: int, *, device=None) -> torch.Tensor:
    """Return H3's native exclusive-cumsum temporal grid at zero origin."""
    if latent_frames <= 0:
        raise ValueError("video latent frame count must be positive")
    spans = torch.tensor(
        [
            FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
            for index in range(latent_frames)
        ],
        dtype=torch.float64,
        device=device,
    )
    return torch.cat((torch.zeros(1, dtype=torch.float64, device=device), spans[:-1].cumsum(0)))


def _chunk_phase_indices(
    dense_latent_frames: int,
    normal_tokens: int,
    single_normal_index: int = 0,
    normal_indices: str = "",
    head_tokens: int = 1,
) -> list[int]:
    """Keep each chunk head plus the requested normal-token phase pattern.

    Two-normal mode retains normal indices 0 and 2.  One-normal mode defaults
    to index 0 for backward compatibility, while allowing experiments to pick
    a different phase such as index 2.
    """
    if dense_latent_frames <= 0:
        raise ValueError("dense latent frame count must be positive")
    if normal_tokens not in (0, 1, 2, 3, 4):
        raise ValueError("chunk_normal_tokens must be between 0 and 4")
    if single_normal_index not in (0, 1, 2, 3):
        raise ValueError("chunk_single_normal_index must be 0, 1, 2, or 3")
    if head_tokens not in (0, 1):
        raise ValueError("chunk_head_tokens must be 0 or 1")
    if normal_indices.strip():
        try:
            phases = [int(value.strip()) for value in normal_indices.split(",")]
        except ValueError as exc:
            raise ValueError("chunk_normal_indices must be comma-separated integers") from exc
        if len(phases) != normal_tokens:
            raise ValueError("chunk_normal_indices count must equal chunk_normal_tokens")
        if len(set(phases)) != len(phases) or any(phase not in range(4) for phase in phases):
            raise ValueError("chunk_normal_indices must be unique values from 0 through 3")
        phases.sort()
    elif normal_tokens == 1:
        phases = [single_normal_index]
    elif normal_tokens == 2:
        phases = [0, 2]
    else:
        phases = list(range(normal_tokens))
    selected: list[int] = []
    for chunk_start in range(0, dense_latent_frames, len(FRAME_PER_TOKEN)):
        if head_tokens:
            selected.append(chunk_start)
        normal = list(
            range(
                chunk_start + 1,
                min(chunk_start + len(FRAME_PER_TOKEN), dense_latent_frames),
            )
        )
        selected.extend(normal[phase] for phase in phases if phase < len(normal))
    return selected


def _compression_dense_indices(
    dense_latent_frames: int, compression_ratio: str
) -> list[int]:
    """Map VPack's public ratio to its fixed native-token schedule."""
    try:
        phases = COMPRESSION_PROFILES[str(compression_ratio)]
    except KeyError as exc:
        choices = ", ".join(COMPRESSION_PROFILES)
        raise ValueError(f"compression_ratio must be one of: {choices}") from exc
    return _chunk_phase_indices(
        dense_latent_frames,
        len(phases),
        normal_indices=",".join(str(phase) for phase in phases),
        head_tokens=1,
    )


def _dense_stride_indices(dense_latent_frames: int, stride: int) -> list[int]:
    """Keep a fixed stride on the native dense grid and always retain the end."""
    if dense_latent_frames <= 0:
        raise ValueError("dense_latent_frames must be positive")
    if stride <= 1:
        raise ValueError("dense stride must be greater than one")
    selected = list(range(0, dense_latent_frames, stride))
    if selected[-1] != dense_latent_frames - 1:
        selected.append(dense_latent_frames - 1)
    return selected


def _chunk_balanced_indices(dense_latent_frames: int, target_tokens: int) -> list[int]:
    """Distribute a target budget while retaining every five-token chunk head."""
    chunks = [
        list(range(start, min(start + len(FRAME_PER_TOKEN), dense_latent_frames)))
        for start in range(0, dense_latent_frames, len(FRAME_PER_TOKEN))
    ]
    if target_tokens < len(chunks) or target_tokens > dense_latent_frames:
        raise ValueError("target_tokens cannot preserve all chunk heads")
    selected = [chunk[0] for chunk in chunks]
    remaining = target_tokens - len(selected)
    # Add the first normal token to every chunk before distributing a second
    # normal token uniformly across time.  This keeps the sparse layout close
    # to the successful 1+2n schedule without front-loading the extra budget.
    for normal_offset in (1, 3, 2, 4):
        candidates = [
            chunk[normal_offset]
            for chunk in chunks
            if len(chunk) > normal_offset and chunk[normal_offset] not in selected
        ]
        if not candidates or remaining <= 0:
            continue
        take = min(remaining, len(candidates))
        if take == len(candidates):
            chosen = candidates
        elif take == 1:
            chosen = [candidates[len(candidates) // 2]]
        else:
            chosen = [
                candidates[round(index * (len(candidates) - 1) / (take - 1))]
                for index in range(take)
            ]
        selected.extend(chosen)
        remaining -= len(chosen)
    if remaining:
        raise RuntimeError("failed to allocate balanced chunk token budget")
    return sorted(selected)


def _reference_layout_segments(layout, refs: list[dict]) -> list[tuple[int, int] | None]:
    """Match each ref block to its visual segment in a native PackedLayout."""
    raw_segments = list(getattr(layout, "segments", ()))
    if not raw_segments or any(
        not isinstance(segment, (tuple, list)) or len(segment) != 3
        for segment in raw_segments
    ):
        raise ComfyH3CompatibilityError(
            "installed ComfyUI uses an unsupported MiniMax-H3 "
            "PackedLayout.segments contract"
        )
    segments = [
        (int(start), int(stop), str(kind)) for start, stop, kind in raw_segments
    ]
    if segments[0][2] != "text" or any(
        start < 0 or stop <= start or (index and start != segments[index - 1][1])
        for index, (start, stop, _kind) in enumerate(segments)
    ):
        raise ComfyH3CompatibilityError(
            "installed ComfyUI produced a non-contiguous MiniMax-H3 segment table"
        )
    index = 1
    while index < len(segments) and segments[index][2] in ("cond", "cond_audio"):
        index += 1

    visual_segments: list[tuple[int, int] | None] = []
    for ref in refs:
        kind = ref.get("kind")
        if kind == "image":
            if index >= len(segments) or segments[index][2] != "ref_img":
                raise RuntimeError("MiniMax-H3 image reference/layout mismatch")
            visual_segments.append((int(segments[index][0]), int(segments[index][1])))
            index += 1
        elif kind == "audio":
            if int(ref.get("ref_audio_t", 0)) > 0:
                if index >= len(segments) or segments[index][2] != "ref_audio":
                    raise RuntimeError("MiniMax-H3 audio reference/layout mismatch")
                index += 1
            visual_segments.append(None)
        elif kind in ("video", "video_audio"):
            if int(ref.get("ref_audio_t", 0)) > 0:
                if index >= len(segments) or segments[index][2] != "ref_audio":
                    raise RuntimeError("MiniMax-H3 video soundtrack/layout mismatch")
                index += 1
            if index >= len(segments) or segments[index][2] != "ref_img":
                raise RuntimeError("MiniMax-H3 video reference/layout mismatch")
            visual_segments.append((int(segments[index][0]), int(segments[index][1])))
            index += 1
        else:
            raise RuntimeError(f"unsupported MiniMax-H3 reference kind {kind!r}")
    return visual_segments


def _apply_physical_time_rope(payload) -> bool:
    """Patch one native Ref2VA layout in place; return whether it was changed."""
    if not isinstance(payload, dict):
        return False
    layout = payload.get("layout")
    refs = payload.get("refs")
    if layout is None or not isinstance(refs, list):
        return False
    if getattr(layout, _LAYOUT_PATCH_MARKER, False):
        return False
    marked = [float(ref.get(ROPE_SCALE_KEY, 1.0)) for ref in refs]
    modes = [str(ref.get(ROPE_MODE_KEY, "affine")) for ref in refs]
    explicit_modes = (
        "chunk_phase",
        "dense_index_chunk_phase",
        "hybrid_tail_dense",
        "partial_tail",
        "dense_stride_3",
        "dense_stride_4",
        "dense_stride_2",
        "chunk_balanced_29",
        "chunk_balanced_32",
        "chunk_balanced_34",
    )
    if not any(
        scale != 1.0 or mode in explicit_modes
        for scale, mode in zip(marked, modes)
    ):
        return False

    visual_segments = _reference_layout_segments(layout, refs)
    position_ids = getattr(layout, "position_ids", None)
    if (
        not torch.is_tensor(position_ids)
        or position_ids.ndim != 2
        or position_ids.shape[1] != 3
        or int(position_ids.shape[0]) != int(layout.segments[-1][1])
    ):
        raise ComfyH3CompatibilityError(
            "installed ComfyUI produced an unsupported MiniMax-H3 position_ids layout"
        )
    corrected = position_ids.clone()
    patched_segments = []
    for ref, scale, mode, segment in zip(refs, marked, modes, visual_segments):
        if (scale == 1.0 and mode not in explicit_modes) or segment is None:
            continue
        if scale <= 0:
            raise ValueError("reference-video RoPE time scale must be positive")
        start, stop = segment
        latent_frames = int(ref["latent_t"])
        temporal = corrected[start:stop, 0]
        if temporal.numel() == 0 or int(torch.unique(temporal).numel()) != latent_frames:
            raise RuntimeError("MiniMax-H3 video reference temporal layout mismatch")
        origin = temporal[0].clone()
        if mode in ("affine", "uniform_time"):
            corrected[start:stop, 0] = origin + (temporal - origin) * scale
            corrected_video_span = _video_rope_span(latent_frames) * scale
            dense_latent_frames = None
            dense_indices = None
        elif mode in ("dense_index", "dense_index_fullspan"):
            dense_latent_frames = int(ref.get(ROPE_DENSE_LATENT_T_KEY, 0))
            if dense_latent_frames < latent_frames:
                raise ValueError(
                    "dense-index RoPE requires dense_latent_t >= compact latent_t"
                )
            dense_indices = torch.round(
                torch.linspace(
                    0,
                    dense_latent_frames - 1,
                    latent_frames,
                    dtype=torch.float64,
                    device=temporal.device,
                )
            ).to(torch.long)
            dense_times = _video_rope_offsets(
                dense_latent_frames, device=temporal.device
            ).index_select(0, dense_indices)
            rows_per_frame, remainder = divmod(temporal.numel(), latent_frames)
            if remainder:
                raise RuntimeError("MiniMax-H3 video rows are not frame-aligned")
            corrected[start:stop, 0] = origin + dense_times.repeat_interleave(
                rows_per_frame
            )
            corrected_video_span = _video_rope_span(dense_latent_frames)
        elif mode in explicit_modes:
            dense_latent_frames = int(ref.get(ROPE_DENSE_LATENT_T_KEY, 0))
            raw_indices = ref.get(ROPE_DENSE_INDICES_KEY)
            if not isinstance(raw_indices, (list, tuple)):
                raise ValueError(f"{mode} RoPE requires explicit dense indices")
            dense_indices = torch.tensor(
                [int(index) for index in raw_indices],
                dtype=torch.long,
                device=temporal.device,
            )
            if (
                dense_indices.numel() != latent_frames
                or dense_latent_frames <= 0
                or int(dense_indices[0]) < 0
                or int(dense_indices[-1]) >= dense_latent_frames
                or bool((dense_indices[1:] <= dense_indices[:-1]).any())
            ):
                raise ValueError(f"invalid {mode} dense index schedule")
            dense_times = _video_rope_offsets(
                dense_latent_frames, device=temporal.device
            ).index_select(0, dense_indices)
            rows_per_frame, remainder = divmod(temporal.numel(), latent_frames)
            if remainder:
                raise RuntimeError("MiniMax-H3 video rows are not frame-aligned")
            corrected[start:stop, 0] = origin + dense_times.repeat_interleave(
                rows_per_frame
            )
            corrected_video_span = _video_rope_span(dense_latent_frames)
        else:
            raise ValueError(f"unknown reference-video RoPE correction mode {mode!r}")

        audio_span = float(ref.get("ref_audio_t", 0))
        video_span = _video_rope_span(latent_frames)
        delta = max(audio_span, corrected_video_span) - max(audio_span, video_span)
        if delta:
            corrected[stop:, 0] += delta
        patched_segments.append(
            {
                "rows": stop - start,
                "latent_frames": latent_frames,
                "mode": mode,
                "scale": scale,
                "dense_latent_frames": dense_latent_frames,
                "dense_index_head": (
                    dense_indices[:8].tolist() if dense_indices is not None else None
                ),
                "corrected_last_t_offset": float(
                    corrected[stop - 1, 0].item() - origin.item()
                ),
                "tail_shift": delta,
            }
        )

    layout.position_ids = corrected
    setattr(layout, _LAYOUT_PATCH_MARKER, True)
    log.info("Ref2VA physical-time RoPE applied: %s", patched_segments)
    return True


def _patch_model_for_physical_time(model):
    if not hasattr(model, "get_model_object"):
        raise ComfyH3CompatibilityError(
            "installed ComfyUI MODEL does not expose get_model_object"
        )
    diffusion_model = model.get_model_object("diffusion_model")
    native_h3 = any(
        model_type.__name__ == "MiniMaxH3Model"
        for model_type in type(diffusion_model).__mro__
    )
    if not native_h3:
        raise ComfyH3CompatibilityError(
            "Ref2VA RoPE correction requires "
            f"MiniMaxH3Model, got {type(diffusion_model).__name__}"
        )
    if not isinstance(getattr(model, "model_options", None), dict):
        raise ComfyH3CompatibilityError(
            "installed ComfyUI MODEL does not expose model_options"
        )
    if not hasattr(model, "clone") or not hasattr(
        model, "set_model_unet_function_wrapper"
    ):
        raise ComfyH3CompatibilityError(
            "installed ComfyUI ModelPatcher does not support composable model wrappers"
        )

    previous_wrapper = model.model_options.get("model_function_wrapper")

    def model_function_wrapper(apply_model, args):
        conditioning = args["c"].copy()
        _apply_physical_time_rope(conditioning.get("minimax_payload"))
        forwarded = args | {"c": conditioning}
        if previous_wrapper is not None:
            return previous_wrapper(apply_model, forwarded)
        return apply_model(forwarded["input"], forwarded["timestep"], **conditioning)

    patched = model.clone()
    patched.set_model_unet_function_wrapper(model_function_wrapper)
    return patched


def _configure_rope_correction(
    model,
    disable: bool,
    condition_fps: float = CONDITION_FPS,
    mode: str = "affine",
    dense_latent_frames: int | None = None,
    chunk_normal_tokens: int = 2,
    explicit_dense_indices: list[int] | None = None,
    chunk_single_normal_index: int = 0,
    chunk_normal_indices: str = "",
    chunk_head_tokens: int = 1,
) -> tuple[object, dict[str, object]]:
    """Return the model and ref metadata for the normal or ablation path."""
    if disable:
        return model, {}
    if condition_fps <= 0:
        raise ValueError("condition_fps must be positive")
    scale = TARGET_FPS / condition_fps
    explicit_modes = (
        "chunk_phase",
        "dense_index_chunk_phase",
        "hybrid_tail_dense",
        "partial_tail",
        "dense_stride_3",
        "dense_stride_4",
        "dense_stride_2",
        "chunk_balanced_29",
        "chunk_balanced_32",
        "chunk_balanced_34",
    )
    if math.isclose(scale, 1.0) and mode not in explicit_modes:
        return model, {}
    if mode not in (
        "affine",
        "uniform_time",
        "dense_index",
        "dense_index_fullspan",
        "chunk_phase",
        "dense_index_chunk_phase",
        "hybrid_tail_dense",
        "partial_tail",
        "dense_stride_3",
        "dense_stride_4",
        "dense_stride_2",
        "chunk_balanced_29",
        "chunk_balanced_32",
        "chunk_balanced_34",
    ):
        raise ValueError(
            "rope_correction_mode must be 'affine', 'uniform_time', "
            "'dense_index', 'dense_index_fullspan', 'chunk_phase', "
            "'dense_index_chunk_phase', "
            "'hybrid_tail_dense', 'partial_tail', 'dense_stride_3', or "
            "'dense_stride_4', 'chunk_balanced_29', 'chunk_balanced_32', or "
            "'chunk_balanced_34'"
        )
    metadata: dict[str, object] = {ROPE_SCALE_KEY: scale, ROPE_MODE_KEY: mode}
    if mode in ("dense_index", "dense_index_fullspan"):
        if dense_latent_frames is None or dense_latent_frames <= 0:
            raise ValueError("dense-index RoPE requires dense_latent_frames")
        metadata[ROPE_DENSE_LATENT_T_KEY] = int(dense_latent_frames)
    elif mode == "chunk_phase":
        if dense_latent_frames is None or dense_latent_frames <= 0:
            raise ValueError("chunk-phase RoPE requires dense_latent_frames")
        metadata[ROPE_DENSE_LATENT_T_KEY] = int(dense_latent_frames)
        metadata[ROPE_DENSE_INDICES_KEY] = _chunk_phase_indices(
            int(dense_latent_frames),
            int(chunk_normal_tokens),
            int(chunk_single_normal_index),
            str(chunk_normal_indices),
            int(chunk_head_tokens),
        )
    elif mode == "dense_index_chunk_phase":
        if dense_latent_frames is None or dense_latent_frames <= 0:
            raise ValueError("dense-index chunk-phase requires dense_latent_frames")
        if not explicit_dense_indices:
            raise ValueError("dense-index chunk-phase requires explicit dense indices")
        metadata[ROPE_DENSE_LATENT_T_KEY] = int(dense_latent_frames)
        metadata[ROPE_DENSE_INDICES_KEY] = [int(index) for index in explicit_dense_indices]
    elif mode in ("hybrid_tail_dense", "partial_tail"):
        if dense_latent_frames is None or dense_latent_frames <= 0:
            raise ValueError(f"{mode} RoPE requires dense_latent_frames")
        if not explicit_dense_indices:
            raise ValueError(f"{mode} RoPE requires explicit_dense_indices")
        metadata[ROPE_DENSE_LATENT_T_KEY] = int(dense_latent_frames)
        metadata[ROPE_DENSE_INDICES_KEY] = [
            int(index) for index in explicit_dense_indices
        ]
    elif mode in ("dense_stride_2", "dense_stride_3", "dense_stride_4"):
        if dense_latent_frames is None or dense_latent_frames <= 0:
            raise ValueError(f"{mode} requires dense_latent_frames")
        stride = int(mode.rsplit("_", 1)[1])
        metadata[ROPE_DENSE_LATENT_T_KEY] = int(dense_latent_frames)
        metadata[ROPE_DENSE_INDICES_KEY] = _dense_stride_indices(
            int(dense_latent_frames), stride
        )
    elif mode in ("chunk_balanced_29", "chunk_balanced_32", "chunk_balanced_34"):
        if dense_latent_frames is None or dense_latent_frames <= 0:
            raise ValueError(f"{mode} requires dense_latent_frames")
        target_tokens = int(mode.rsplit("_", 1)[1])
        metadata[ROPE_DENSE_LATENT_T_KEY] = int(dense_latent_frames)
        metadata[ROPE_DENSE_INDICES_KEY] = _chunk_balanced_indices(
            int(dense_latent_frames), target_tokens
        )
    return _patch_model_for_physical_time(model), metadata


def _crop_audio(audio: dict, duration: float) -> dict:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    return {
        "waveform": waveform[..., : int(duration * sample_rate)],
        "sample_rate": sample_rate,
    }


class VPack:
    """Pack Ref2VA reference-video latent frames without changing size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "compression_ratio": (
                    list(COMPRESSION_PROFILES),
                    {
                        "default": "2/5",
                        "tooltip": "Reference-video compression level.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "pack"
    CATEGORY = "model/conditioning/minimax"
    DESCRIPTION = "Compress reference-video conditioning for MiniMax H3 Ref2VA."

    def pack(self, model, positive, compression_ratio):
        if compression_ratio not in COMPRESSION_PROFILES:
            choices = ", ".join(COMPRESSION_PROFILES)
            raise ValueError(f"compression_ratio must be one of: {choices}")
        if compression_ratio == "0/5":
            return model, positive

        packed_conditioning = []
        packed_video_count = 0
        for embedding, metadata in positive:
            refs = metadata.get("minimax_refs")
            if not isinstance(refs, list):
                packed_conditioning.append([embedding, metadata])
                continue

            packed_refs = []
            for ref in refs:
                if ref.get("kind") not in ("video", "video_audio"):
                    packed_refs.append(ref)
                    continue
                video_latent = ref.get("latent")
                dense_t = int(ref.get("latent_t", 0))
                if (
                    not torch.is_tensor(video_latent)
                    or video_latent.ndim != 5
                    or dense_t != int(video_latent.shape[2])
                ):
                    raise ValueError("Ref2VA reference video has an invalid latent")
                dense_indices = _compression_dense_indices(dense_t, compression_ratio)
                selected = torch.tensor(
                    dense_indices, dtype=torch.long, device=video_latent.device
                )
                packed_ref = ref.copy()
                packed_ref["latent"] = video_latent.index_select(2, selected)
                packed_ref["latent_t"] = len(dense_indices)
                packed_ref.update({
                    ROPE_SCALE_KEY: 1.0,
                    ROPE_MODE_KEY: "chunk_phase",
                    ROPE_DENSE_LATENT_T_KEY: dense_t,
                    ROPE_DENSE_INDICES_KEY: dense_indices,
                })
                packed_refs.append(packed_ref)
                packed_video_count += 1

            packed_metadata = metadata.copy()
            packed_metadata["minimax_refs"] = packed_refs
            packed_conditioning.append([embedding, packed_metadata])

        if packed_video_count == 0:
            raise ValueError(
                "VPack requires MiniMax H3 Ref2VA conditioning "
                "with at least one encoded reference video"
            )
        log.info(
            "VPack packed %d reference video(s) at ratio %s",
            packed_video_count,
            compression_ratio,
        )
        return _patch_model_for_physical_time(model), packed_conditioning


SparkVPack = VPack  # Keep workflows saved with the original node ID loadable.

NODE_CLASS_MAPPINGS = {"VPack": VPack, "SparkVPack": VPack}

NODE_DISPLAY_NAME_MAPPINGS = {"VPack": "VPack", "SparkVPack": "VPack"}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "VPack",
    "SparkVPack",
]
