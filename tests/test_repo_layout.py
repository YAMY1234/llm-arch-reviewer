from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def audited_model_ids() -> set[str]:
    return {
        f"{path.name}_v2"
        for path in (REPO_ROOT / "catalog").iterdir()
        if path.is_dir() and (path / "model_ir.yaml").is_file()
    }


def test_public_site_has_one_shared_viewer() -> None:
    viewers = sorted((REPO_ROOT / "docs").glob("*viewer*.html"))
    assert viewers == [REPO_ROOT / "docs" / "viewer.html"]


def test_public_model_bundles_exactly_match_audited_catalogs() -> None:
    bundle_dirs = {
        path.name
        for path in (REPO_ROOT / "docs").iterdir()
        if path.is_dir() and (path / "arch_data.json").is_file()
    }
    assert bundle_dirs == audited_model_ids()


def test_legacy_dsv4_pipeline_is_removed() -> None:
    assert not (REPO_ROOT / "models" / "dsv4").exists()
    assert not (REPO_ROOT / "docs" / "dsv4").exists()
    assert "model=dsv4" not in (REPO_ROOT / "README.md").read_text()
    assert "model=dsv4" not in (REPO_ROOT / "docs" / "index.html").read_text()


def test_qwen38_flash_next_uses_official_identity_with_legacy_url_alias() -> None:
    catalog_root = REPO_ROOT / "catalog" / "qwen38_flash_next"
    model_ir = (catalog_root / "model_ir.yaml").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    landing = (REPO_ROOT / "docs" / "index.html").read_text()
    viewer = (REPO_ROOT / "docs" / "viewer.html").read_text()

    assert catalog_root.is_dir()
    assert not (REPO_ROOT / "catalog" / "qwen40").exists()
    assert "model_id: qwen38_flash_next" in model_ir
    assert "model_label: Qwen3.8-Flash-Next BF16" in model_ir
    assert "Qwen 4.0 Air Example" not in model_ir
    assert "model=qwen38_flash_next_v2" in readme
    assert "model=qwen38_flash_next_v2" in landing
    assert 'qwen40_v2: "qwen38_flash_next_v2"' in viewer
    assert (REPO_ROOT / "docs" / "qwen38_flash_next_v2").is_dir()
    assert not (REPO_ROOT / "docs" / "qwen40_v2").exists()
