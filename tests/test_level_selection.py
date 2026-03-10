from types import SimpleNamespace

from wsi_seg.slide import OpenSlideReader, SlideLevel


def _reader_with_levels() -> OpenSlideReader:
    reader = OpenSlideReader.__new__(OpenSlideReader)
    reader.metadata = SimpleNamespace(
        levels=[
            SlideLevel(index=0, width=1, height=1, downsample=1.0, mpp_x=0.2425, mpp_y=0.2426),
            SlideLevel(index=1, width=1, height=1, downsample=2.0, mpp_x=0.4850, mpp_y=0.4853),
            SlideLevel(index=2, width=1, height=1, downsample=4.0, mpp_x=0.9701, mpp_y=0.9706),
        ]
    )
    return reader


def test_choose_level_nearest_policy_picks_closest_native_level() -> None:
    reader = _reader_with_levels()
    selection = reader.choose_level(0.88, policy="nearest")
    assert selection.level == 2
    assert selection.policy == "nearest"


def test_choose_level_prefer_higher_picks_finer_native_level() -> None:
    reader = _reader_with_levels()
    selection = reader.choose_level(0.88, policy="prefer_higher")
    assert selection.level == 1
    assert selection.policy == "prefer_higher"


def test_choose_level_prefer_higher_bounded_falls_back_when_oversample_too_large() -> None:
    reader = _reader_with_levels()
    selection = reader.choose_level(
        0.88,
        policy="prefer_higher_bounded",
        max_native_oversample_factor=1.5,
    )
    assert selection.level == 2


def test_choose_level_prefer_higher_bounded_accepts_finer_level_within_cap() -> None:
    reader = _reader_with_levels()
    selection = reader.choose_level(
        0.88,
        policy="prefer_higher_bounded",
        max_native_oversample_factor=2.0,
    )
    assert selection.level == 1
