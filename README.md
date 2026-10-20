# Spark-VPack

Spark-VPack is a ComfyUI node that compresses reference-video conditioning in MiniMax H3 Ref2VA workflows. It can be used with any compatible workflow that supplies an encoded reference video.

## Install

Requires ComfyUI 0.38.0 or newer with MiniMax H3 Ref2VA support and Python 3.10 or newer. Extract `Spark-VPack` into `ComfyUI/custom_nodes/`, then restart ComfyUI.

## Find the node

Double-click empty canvas space, search for `Spark-VPack`, and select it. You can also find it under `model → conditioning → minimax` in the canvas menu.

## Connect

1. Use a MiniMax H3 Ref2VA workflow with at least one encoded reference video. Its `positive` conditioning must contain the reference video in `minimax_refs`.
2. Connect that `positive` output and the MiniMax H3 `MODEL` to **Spark-VPack**.
3. Connect Spark-VPack's `model` and `positive` outputs to the usual scheduler and guider. Keep the generated `latent` connected directly to the sampler.

## Compression settings

**Use `2/5` as the recommended starting point; it is also the default.** In our 1344×768, approximately 4-second, 4-step ikun Ref2VA test, it preserved the character and scene while shortening the four-step DiT sampling segment by about 24% versus `0/5` on SM120. This is a single quick test, so the result will vary with the workflow and hardware.

Choose `compression_ratio` to control how strongly the reference video is compressed.

| Setting | Use |
| --- | --- |
| `0/5` | No compression; baseline |
| `1/5` | Lighter compression |
| `2/5` | Recommended balance of consistency and speed |
| `3/5` | Stronger compression |

The matched video and timing comparison currently covers only `0/5` and `2/5`; equivalent quality and speed results for `1/5` and `3/5` are pending.

Reference-image and reference-video resolution, source frame rate, generation resolution, and generation length are handled by the upstream workflow.

## ❤️ Special Thanks

Thanks to [Smthem](https://space.bilibili.com/439655233), [AI绘画KK](https://space.bilibili.com/313528845), [AIWood爱屋研究院](https://space.bilibili.com/503934057), 激进派ikun, and friends across the open-source community for sharing workflows and materials that helped us carry out the initial validation of VPack.
