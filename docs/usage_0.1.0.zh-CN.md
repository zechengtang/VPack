# Spark-VPack 0.1.0 使用说明

Spark-VPack 用于带参考视频的 MiniMax H3 Reference to Video（Ref2VA）工作流。

## 导入节点

需要支持 MiniMax H3 Ref2VA 的 ComfyUI 0.38.0 或更新版本，以及 Python 3.10 或更新版本。

1. 解压安装包，得到 `Spark-VPack` 文件夹。
2. 将整个文件夹放到当前 ComfyUI 的 `custom_nodes` 目录中，最终路径为 `ComfyUI/custom_nodes/Spark-VPack/__init__.py`。
3. 重启 ComfyUI。在画布空白处**右键 → Add Node → model → conditioning → minimax → Spark-VPack**，逐级点选添加；也可以双击画布空白处，搜索 `Spark-VPack`。

## 接入 Ref2VA 工作流

1. 在 MiniMax H3 Ref2VA 工作流中准备至少一个已编码参考视频；`positive` 条件中应包含对应的 `minimax_refs`。
2. 将该 `positive` 输出接到 Spark-VPack 的 `positive` 输入；将 MiniMax H3 的 `MODEL` 接到 Spark-VPack 的 `model` 输入。
3. 将 Spark-VPack 输出的 `positive` 和 `model` 接回后续的引导器、采样器等节点。生成用 `latent` 仍直接接到采样器。

`compression_ratio` 是参考视频 latent 的压缩比例。
