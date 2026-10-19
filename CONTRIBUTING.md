# Development and release

Spark-VPack is an independent ComfyUI custom node. The repository root is the
node source directory; it can be placed directly at
`ComfyUI/custom_nodes/Spark-VPack/`. It does not import Spark-H3.

Run the source and packaging tests with the Python used by a compatible
ComfyUI installation:

```bash
python -m pytest -q tests
```

Build a standalone installation archive:

```bash
python tools/build_package.py
```

The command writes `dist/Spark-VPack-<version>.zip`. The archive contains only
the files needed by the ComfyUI node. Update the version in `pyproject.toml`
before building a new release, then verify node registration in the target
ComfyUI environment. The v0.1.0 release notes are in `docs/`.
