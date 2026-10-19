# Spark-VPack

Spark-VPack is a small ComfyUI node for MiniMax H3 Ref2VA workflows with
reference video.
It drops selected **temporal latent frames** from each five-frame reference
chunk. It does not resize, crop, decode, or encode images or video, and it does
not change the generated latent.

## Install

Requires ComfyUI 0.38.0 or newer with MiniMax H3 Ref2VA support and
Python 3.10 or newer. Extract `Spark-VPack` into `ComfyUI/custom_nodes/`, then
restart ComfyUI.

## Find the node

Double-click empty canvas space, search for `Spark-VPack`, and select it.
You can also find it under `model → conditioning → minimax` in the canvas menu.

## Connect

1. Use a MiniMax H3 Ref2VA workflow with at least one encoded reference video.
   Its `positive` conditioning must contain the reference video in `minimax_refs`.
2. Connect that `positive` output and the MiniMax H3 `MODEL` to **Spark-VPack**.
3. Connect Spark-VPack's `model` and `positive` outputs to the usual scheduler
   and guider. Keep the generated `latent` connected directly to the sampler.

## Compression settings

**Use `2/5` as the recommended starting point; it is also the default.** In
our 1344×768, approximately 4-second, 4-step ikun Ref2VA test, it preserved
the character and scene while shortening the four-step DiT sampling segment
by about 24% versus `0/5` on SM120. This is a single quick test, so the
result will vary with the workflow and hardware.

`compression_ratio` means the number of **reference-video latent temporal
frames** dropped from each five-frame chunk. The first frame of every chunk
is retained.

| Setting | Frames kept per chunk | Use |
| --- | ---: | --- |
| `0/5` | 5 | No compression; baseline |
| `1/5` | 4 | Lighter compression |
| `2/5` | 3 | Recommended balance of consistency and speed |
| `3/5` | 2 | Stronger compression |

The matched video and timing comparison currently covers only `0/5` and
`2/5`; equivalent quality and speed results for `1/5` and `3/5` are pending.

Reference-image and reference-video resolution, source frame rate, generation
resolution, and generation length are handled by the upstream workflow.
