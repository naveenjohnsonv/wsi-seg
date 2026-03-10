from wsi_seg.config import AppConfig
from wsi_seg.pipeline import exports_requested, needs_materialized_export_mask

BASE = {
    "model": {
        "target_mpp": 0.88,
        "patch_px": 512,
        "stride_px": 384,
        "halo_px": 64,
        "batch_size": 4,
        "threshold": 0.5,
    },
}


def test_exports_requested_false_when_all_outputs_disabled() -> None:
    cfg = AppConfig.model_validate({
        **BASE,
        "output": {
            "write_tiff": False,
            "write_ome_tiff": False,
            "write_previews": False,
        },
    })
    assert exports_requested(cfg) is False
    assert needs_materialized_export_mask(cfg) is False


def test_needs_materialized_export_mask_only_for_tiff_artifacts() -> None:
    previews_only = AppConfig.model_validate({
        **BASE,
        "output": {
            "write_tiff": False,
            "write_ome_tiff": False,
            "write_previews": True,
        },
    })
    assert exports_requested(previews_only) is True
    assert needs_materialized_export_mask(previews_only) is False

    ome_export = AppConfig.model_validate({
        **BASE,
        "output": {
            "write_tiff": False,
            "write_ome_tiff": True,
            "write_previews": False,
        },
    })
    assert exports_requested(ome_export) is True
    assert needs_materialized_export_mask(ome_export) is True
