# Spark-VPack 0.1.0 发布说明

Spark-VPack 对 ComfyUI 原生 MiniMax H3 Ref2VA 任务中的参考视频进行压缩，缩短 DiT 处理的序列。支持 ComfyUI 0.38.0 及更新版本。

| 压缩比例 | 说明 |
| --- | --- |
| `0/5` | 不压缩 |
| `1/5` | 轻度压缩 |
| `2/5` | **推荐，默认值** |
| `3/5` | 更强压缩 |

推荐 `2/5`：在 SM120 的 4 步 768p ikun 测试中，人物与场景基本保持，DiT 采样耗时比 `0/5` 约减少 24%。这是单次测试；`1/5`、`3/5` 尚无同条件对比结果。

## 使用方式

将 `Spark-VPack` 放入 `ComfyUI/custom_nodes/` 并重启。把原生 Ref2VA 的 `positive` 和模型 `MODEL` 接入 Spark-VPack，再将输出接回采样工作流；生成用 latent 仍直接接采样器。
