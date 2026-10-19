# Spark-VPack 0.1.0 Release Notes

Spark-VPack is an independent ComfyUI node for compressing the reference-video tokens used by the native MiniMax H3 Ref2VA task. It does not require Spark-H3. You can use it on its own or alongside Sol-Attn or Veda. It supports ComfyUI 0.38.0 and newer.

| Compression ratio | Description |
| --- | --- |
| `0/5` | 0% compression |
| `1/5` | 20% compression |
| `2/5` | **40% compression; recommended default** |
| `3/5` | 60% compression |

The ratio describes compression of the Ref2VA reference-video tokens. Actual reduction can vary slightly with video length.

## Usage

Place the `Spark-VPack` folder in `ComfyUI/custom_nodes/` and restart ComfyUI. Connect the native Ref2VA node's `positive` output and the model's `MODEL` output to Spark-VPack, then connect its outputs back into the sampling workflow. Keep the generation latent connected directly to the sampler.
