# Spark-VPack

Spark-VPack 是供带参考视频的 MiniMax H3 Ref2VA 工作流使用的轻量 ComfyUI 节点。它只在参考视频 latent 的每五帧分组中按档位丢弃时间帧，不缩放、裁剪、解码或编码图像与视频，也不修改生成用的 latent。

## 安装

需要支持 MiniMax H3 Ref2VA 的 ComfyUI 0.38.0 或更新版本，以及 Python 3.10 或更新版本。将 `Spark-VPack` 解压到 `ComfyUI/custom_nodes/`，然后重启 ComfyUI。

## 在界面中找到节点

在 ComfyUI 画布空白处双击，输入 `Spark-VPack`，点击搜索结果即可添加。也可以右键画布，在 `model → conditioning → minimax` 分类下找到它。

## 连接方式

1. 使用包含至少一个已编码参考视频的 MiniMax H3 Ref2VA 工作流；其 `positive` 条件中应通过 `minimax_refs` 提供参考视频。
2. 将该 `positive` 输出和 MiniMax H3 `MODEL` 接入 **Spark-VPack**。
3. 将 Spark-VPack 的 `model`、`positive` 输出接入常规调度器和引导器；生成用的 `latent` 仍直接接入采样器。

## 压缩档位

**推荐 `2/5`，也是默认值。** 在目前的 1344×768、约 4 秒、4 步 ikun Ref2VA 测试中，它保持了人物和场景的一致性，且在 SM120 上的四步 DiT 采样段比 `0/5` 短约 24%。这是单次快速测试结果，实际收益会随工作流和硬件变化。

`compression_ratio` 表示每组五个**参考视频 latent 时间帧**中丢弃的帧数；每组首帧始终保留。可选项为：

| 档位 | 每组保留 | 用途 |
| --- | --- | --- |
| `0/5` | 5 帧 | 不压缩，作为原始参考基线 |
| `1/5` | 4 帧 | 较轻的压缩 |
| `2/5` | 3 帧 | 推荐，兼顾一致性与速度 |
| `3/5` | 2 帧 | 更强的压缩 |

目前同条件视频与耗时对比只覆盖 `0/5` 和 `2/5`；`1/5`、`3/5` 尚未得到相同测试的画质与速度结论。

参考图像和参考视频分辨率、输入帧率、生成分辨率与生成时长由上游工作流处理。
