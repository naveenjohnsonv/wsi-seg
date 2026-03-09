from wsi_seg.pipeline import axis_positions, valid_crop_bounds


def test_axis_positions_covers_tail() -> None:
    positions = axis_positions(length=1000, patch_px=512, stride_px=384)
    assert positions == [0, 384, 488]


def test_axis_positions_small_image() -> None:
    assert axis_positions(length=256, patch_px=512, stride_px=384) == [0]


def test_valid_crop_bounds_inner_tile() -> None:
    left, top, right, bottom = valid_crop_bounds(
        out_x=384,
        out_y=384,
        patch_px=512,
        out_w=1200,
        out_h=1200,
        halo_px=64,
    )
    assert (left, top, right, bottom) == (64, 64, 448, 448)


def test_valid_crop_bounds_top_left_edge() -> None:
    left, top, right, bottom = valid_crop_bounds(
        out_x=0,
        out_y=0,
        patch_px=512,
        out_w=1200,
        out_h=1200,
        halo_px=64,
    )
    assert (left, top, right, bottom) == (0, 0, 448, 448)
