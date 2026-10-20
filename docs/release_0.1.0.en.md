# VPack 0.1.0 Release Notes

VPack is an independent ComfyUI node for compressing reference-video conditioning in MiniMax H3 Ref2VA workflows. It does not require Spark-H3. You can use it on its own or alongside Sol-Attn or Veda. It supports ComfyUI 0.38.0 and newer. The input conditioning must contain an encoded reference video in `minimax_refs`.

| Compression ratio | Description |
| --- | --- |
| `0/5` | 0% compression |
| `1/5` | 20% compression |
| `2/5` | **40% compression; recommended default** |
| `3/5` | 60% compression |

Higher settings apply stronger reference-video compression. Actual reduction can vary slightly with video length.

## Usage

Place the `VPack` folder in `ComfyUI/custom_nodes/` and restart ComfyUI. Connect the Ref2VA workflow's `positive` conditioning and MiniMax H3 `MODEL` to VPack, then connect its outputs back into the sampling workflow. Keep the generation latent connected directly to the sampler.
