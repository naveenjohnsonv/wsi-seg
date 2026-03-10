from types import SimpleNamespace

from wsi_seg.slide import OpenSlideReader


def test_extract_mpp_from_direct_keys() -> None:
    mpp_x, mpp_y, source = OpenSlideReader._extract_mpp_from_properties(
        {"openslide.mpp-x": "0.24", "openslide.mpp-y": "0.25"}
    )
    assert (mpp_x, mpp_y) == (0.24, 0.25)
    assert source == "openslide.mpp-x/y"


def test_extract_mpp_from_aperio_key() -> None:
    mpp_x, mpp_y, source = OpenSlideReader._extract_mpp_from_properties({"aperio.MPP": "0.50"})
    assert (mpp_x, mpp_y) == (0.5, 0.5)
    assert source == "aperio.MPP"


def test_extract_mpp_from_text_description() -> None:
    props = {"tiff.ImageDescription": "scanner metadata ... microns per pixel = 0.88"}
    mpp_x, mpp_y, source = OpenSlideReader._extract_mpp_from_properties(props)
    assert (mpp_x, mpp_y) == (0.88, 0.88)
    assert source.startswith("text:")


def test_extract_mpp_from_tiff_resolution() -> None:
    props = {
        "tiff.XResolution": "10000",
        "tiff.YResolution": "10000",
        "tiff.ResolutionUnit": "centimeter",
    }
    mpp_x, mpp_y, source = OpenSlideReader._extract_mpp_from_properties(props)
    assert round(mpp_x, 6) == 1.0
    assert round(mpp_y, 6) == 1.0
    assert source == "tiff.resolution"


def test_extract_mpp_from_objective_power_estimate() -> None:
    props = {"openslide.objective-power": "20"}
    mpp_x, mpp_y, source = OpenSlideReader._extract_mpp_from_properties(props)
    assert (mpp_x, mpp_y) == (0.5, 0.5)
    assert source == "objective-power-estimate"


def test_mpp_override_fills_missing_axis() -> None:
    mpp_x, mpp_y, source = OpenSlideReader._override_mpp(0.75, None)
    assert (mpp_x, mpp_y) == (0.75, 0.75)
    assert source == "config.override"


def test_output_frame_uses_bounds_as_native_region() -> None:
    reader = OpenSlideReader.__new__(OpenSlideReader)
    reader.metadata = SimpleNamespace(
        width=1000,
        height=800,
        mpp_x=0.5,
        mpp_y=0.5,
        bounds=SimpleNamespace(x=200, y=100, width=400, height=300),
    )
    frame = reader.output_frame(1.0, use_bounds=True)
    assert (frame.origin_x_level0, frame.origin_y_level0) == (200, 100)
    assert (frame.width_level0, frame.height_level0) == (400, 300)
    assert (frame.out_w, frame.out_h) == (200, 150)
    assert frame.actual_output_mpp_x == 1.0
    assert frame.actual_output_mpp_y == 1.0
