from __future__ import annotations

import importlib.util
import sys
import zipfile

from tools.build_package import PACKAGE_NAME, assemble, make_zip


def test_independent_package_contents_and_entry_point(tmp_path):
    package = assemble(tmp_path, version="0.1.0", publisher_id="test-publisher")
    assert {path.name for path in package.iterdir()} == {
        "LICENSE",
        "README.md",
        "README.zh-CN.md",
        "__init__.py",
        "comfyui_ref2va.py",
        "comfyui_ref2va_compat.py",
        "pyproject.toml",
    }
    assert not (package / "comfyui_nodes.py").exists()
    assert not (package / "install.py").exists()
    assert not (package / "web").exists()
    assert not list(package.rglob("*.whl"))

    module_name = "spark_vpack_test_package"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    assert set(module.NODE_CLASS_MAPPINGS) == {"SparkVPack"}
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {"SparkVPack": "Spark-VPack"}


def test_independent_metadata_and_documentation(tmp_path):
    package = assemble(tmp_path, version="1.2.3", publisher_id="publisher")
    metadata = (package / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "comfyui-spark-vpack"' in metadata
    assert 'version = "1.2.3"' in metadata
    assert 'PublisherId = "publisher"' in metadata
    assert 'DisplayName = "Spark-VPack"' in metadata
    assert 'requires-comfyui = ">=0.38.0"' in metadata
    assert "Operating System :: OS Independent" in metadata

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            package / "README.md",
            package / "README.zh-CN.md",
            package / "comfyui_ref2va.py",
            package / "comfyui_ref2va_compat.py",
        )
    )
    assert "spark-vpack" in public_text.casefold()
    readme_text = "\n".join(
        (package / name).read_text(encoding="utf-8")
        for name in ("README.md", "README.zh-CN.md")
    )
    assert "condition_fps" not in readme_text
    assert "rope_correction_mode" not in readme_text


def test_independent_archive_has_one_installable_root(tmp_path):
    package = assemble(tmp_path, version="0.1.0", publisher_id="publisher")
    archive = make_zip(package, tmp_path, version="0.1.0")
    assert archive.name == f"{PACKAGE_NAME}-0.1.0.zip"
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert names
    assert all(name.startswith(f"{PACKAGE_NAME}/") for name in names)
    assert f"{PACKAGE_NAME}/__init__.py" in names
