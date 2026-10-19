"""Build a standalone Spark-VPack custom node archive from this repository."""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "Spark-VPack"
PACKAGE_FILES = (
    "LICENSE",
    "README.md",
    "README.zh-CN.md",
    "__init__.py",
    "comfyui_ref2va.py",
    "comfyui_ref2va_compat.py",
)


def _metadata_value(key: str) -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(rf'(?m)^{re.escape(key)} = "([^"]+)"$', text)
    if match is None:
        raise ValueError(f"missing {key} in pyproject.toml")
    return match.group(1)


def assemble(output_dir: Path, *, version: str, publisher_id: str) -> Path:
    """Create one installable directory without tests or repository tooling."""
    target = output_dir / PACKAGE_NAME
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in PACKAGE_FILES:
        shutil.copy2(ROOT / name, target / name)
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    metadata = re.sub(
        r'(?m)^version = "[^"]+"$', f'version = "{version}"', metadata, count=1
    )
    metadata = re.sub(
        r'(?m)^PublisherId = "[^"]+"$',
        f'PublisherId = "{publisher_id}"',
        metadata,
        count=1,
    )
    (target / "pyproject.toml").write_text(metadata, encoding="utf-8")
    return target


def make_zip(package_dir: Path, output_dir: Path, *, version: str) -> Path:
    """Create a deterministic ZIP with a single Spark-VPack root."""
    archive = output_dir / f"{PACKAGE_NAME}-{version}.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(PACKAGE_NAME) / path.relative_to(package_dir)
            info = zipfile.ZipInfo(
                relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes())
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--version", default=_metadata_value("version"))
    parser.add_argument("--publisher-id", default=_metadata_value("PublisherId"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = assemble(
        args.output_dir, version=args.version, publisher_id=args.publisher_id
    )
    print(package)
    print(make_zip(package, args.output_dir, version=args.version))


if __name__ == "__main__":
    main()
