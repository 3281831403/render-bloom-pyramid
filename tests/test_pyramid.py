import pytest

from bloom.pyramid import (
    BloomError,
    apply_bloom,
    bloom_batch,
    build_pyramid,
    combine,
    downsample,
    threshold,
    upsample,
    validate_pyramid,
)


def _image(height, width, value=0.0):
    return [[float(value) for _ in range(width)] for _ in range(height)]


def _assert_close(actual, expected, tolerance=1e-6):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert actual_value == pytest.approx(expected_value, abs=tolerance)


def _assert_shape(image, height, width):
    assert len(image) == height
    assert all(len(row) == width for row in image)


def test_threshold_uses_soft_knee():
    img = [
        [0.35, 0.40, 0.50],
        [0.60, 0.80, 0.90],
    ]

    out = threshold(img, 0.60, 0.20)

    _assert_shape(out, 2, 3)
    assert out[0][0] == pytest.approx(0.0, abs=1e-6)
    assert out[0][1] == pytest.approx(0.0, abs=1e-6)
    assert out[0][2] == pytest.approx(0.0125, abs=1e-6)
    assert out[1][0] == pytest.approx(0.05, abs=1e-6)
    assert out[1][1] == pytest.approx(0.20, abs=1e-6)
    assert out[1][2] == pytest.approx(0.30, abs=1e-6)


def test_downsample_odd_size_keeps_last_column():
    img = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    out = downsample(img)

    _assert_shape(out, 2, 2)
    assert out[0][0] == pytest.approx(0.30, abs=1e-6)
    assert out[0][1] == pytest.approx(0.45, abs=1e-6)
    assert out[1][0] == pytest.approx(0.75, abs=1e-6)
    assert out[1][1] == pytest.approx(0.90, abs=1e-6)


def test_upsample_uses_bilinear_sampling():
    img = [
        [0.0, 0.4],
        [0.2, 0.6],
    ]

    out = upsample(img, (3, 3))

    _assert_shape(out, 3, 3)
    assert out[0][0] == pytest.approx(0.0, abs=1e-6)
    assert out[0][2] == pytest.approx(0.4, abs=1e-6)
    assert out[2][0] == pytest.approx(0.2, abs=1e-6)
    assert out[2][2] == pytest.approx(0.6, abs=1e-6)
    assert out[0][1] == pytest.approx(0.2, abs=1e-6)
    assert out[1][0] == pytest.approx(0.1, abs=1e-6)
    assert out[1][1] == pytest.approx(0.3, abs=1e-6)
    assert out[1][2] == pytest.approx(0.5, abs=1e-6)


def test_build_pyramid_rejects_too_many_levels():
    img = _image(4, 4, 0.5)

    with pytest.raises(BloomError):
        build_pyramid(img, 4)


def test_combine_resamples_layers_to_base_size():
    base = _image(4, 4, 0.2)
    coarse = _image(2, 2, 0.6)

    out = combine([base, coarse], [0.5, 0.5])

    _assert_shape(out, 4, 4)
    for row in out:
        for value in row:
            assert value == pytest.approx(0.4, abs=1e-6)


def test_combine_normalizes_weights():
    shallow = _image(2, 2, 0.2)
    deep = _image(2, 2, 0.4)

    out = combine([shallow, deep], [3.0, 1.0])

    for row in out:
        for value in row:
            assert value == pytest.approx(0.25, abs=1e-6)
            assert value < 1.0


def test_apply_bloom_clamps_strength():
    img = _image(2, 2, 0.9)

    overdriven = apply_bloom(img, 0.1, 1, 5.0)
    reference = apply_bloom(img, 0.1, 1, 1.0)
    negative = apply_bloom(img, 0.1, 1, -2.0)

    assert all(0.0 <= value <= 1.0 + 1e-6 for row in overdriven for value in row)
    _assert_close(overdriven, reference)
    _assert_close(negative, img)


def test_validate_pyramid_flags_non_decreasing_sizes():
    bad = [_image(4, 4, 0.1), _image(4, 4, 0.2)]

    issues = validate_pyramid(bad)

    assert issues
    assert all(isinstance(issue, str) and issue.strip() for issue in issues)


def test_bloom_batch_isolates_images_and_rolls_back():
    config = {"threshold": 0.35, "levels": 2, "strength": 0.8}
    images = [_image(3, 3, 0.2), _image(3, 3, 0.8)]

    result = bloom_batch(images, config)

    assert result["ok"] is True
    assert result["errors"] == []
    assert len(result["results"]) == 2
    _assert_close(result["results"][0], apply_bloom(images[0], **config))
    _assert_close(result["results"][1], apply_bloom(images[1], **config))
    assert result["results"][0] != result["results"][1]

    ragged = [[0.1, 0.2], [0.3]]
    rolled_back = bloom_batch([images[0], ragged], config)

    assert rolled_back["ok"] is False
    assert rolled_back["results"] == []
    assert rolled_back["errors"]


def test_black_image_short_circuits_to_exact_zero():
    black = _image(3, 4, 0.0)

    out = apply_bloom(black, 0.2, 3, 0.7)

    _assert_shape(out, 3, 4)
    for row in out:
        for value in row:
            assert value == 0.0


def test_zero_strength_preserves_input():
    img = [
        [0.0, 0.2, 0.5],
        [0.7, 0.9, 1.0],
    ]

    out = apply_bloom(img, 0.3, 2, 0.0)

    _assert_close(out, img)


def test_batch_of_black_images_stays_zero():
    black = _image(2, 3, 0.0)

    result = bloom_batch(
        [black, black],
        {"threshold": 0.1, "levels": 2, "strength": 1.0},
    )

    assert result["ok"] is True
    assert len(result["results"]) == 2
    for image in result["results"]:
        _assert_shape(image, 2, 3)
        for row in image:
            for value in row:
                assert value == 0.0


def test_zero_levels_raises_bloom_error():
    with pytest.raises(BloomError):
        apply_bloom(_image(2, 2, 0.5), 0.2, 0, 0.5)


def test_single_level_pyramid_matches_source_size():
    img = _image(3, 5, 0.25)

    pyramid = build_pyramid(img, 1)

    assert isinstance(pyramid, list)
    assert len(pyramid) == 1
    _assert_shape(pyramid[0], 3, 5)


def test_batch_success_preserves_order():
    config = {"threshold": 0.3, "levels": 2, "strength": 0.5}
    images = [
        _image(2, 2, 0.15),
        _image(2, 2, 0.55),
        _image(2, 2, 0.85),
    ]

    result = bloom_batch(images, config)

    assert result["ok"] is True
    assert len(result["results"]) == 3
    expected = [apply_bloom(image, **config) for image in images]
    for actual, wanted in zip(result["results"], expected):
        _assert_close(actual, wanted)
