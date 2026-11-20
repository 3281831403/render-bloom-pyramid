"""Core building blocks for a linear-light bloom pyramid.

Every image is a non-empty, rectangular ``list[list[float]]`` in row-major
order. Pixel values are linear-light grayscale in ``[0.0, 1.0]``: no gamma
conversion is performed, and all arithmetic is linear. Structural and range
violations raise :class:`BloomError`. Numeric results are expected to match
within an absolute tolerance of ``1e-6`` unless a contract explicitly requires
an exact value.
"""

import math


class BloomError(ValueError):
    """Raised when an image, pyramid, or parameter violates its contract."""


def _validate_image(img, name="img"):
    """Check the common image contract, returning ``(height, width)``."""
    if not isinstance(img, (list, tuple)) or len(img) == 0:
        raise BloomError(f"{name} must be a non-empty list of rows")
    width = None
    for row in img:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise BloomError(f"{name} rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise BloomError(f"{name} must be rectangular (ragged rows)")
        for value in row:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise BloomError(f"{name} pixels must be real numbers")
            if not math.isfinite(value):
                raise BloomError(f"{name} pixels must be finite")
            if value < 0.0 or value > 1.0:
                raise BloomError(f"{name} pixels must lie in [0.0, 1.0]")
    return len(img), width


def _copy_image(img):
    return [list(row) for row in img]


def threshold(img: list[list[float]], threshold: float, knee: float) -> list[list[float]]:
    """Apply a linear-light soft threshold and return a new image.

    ``img`` must be a non-empty rectangular image with finite values in
    ``[0.0, 1.0]``. ``threshold`` is the knee centre in ``[0.0, 1.0]`` and
    ``knee`` is the half-width of the transition, with
    ``0.0 <= knee <= threshold``. Also accept ``threshold == knee == 0``.

    For each pixel value ``v``, let ``lo = threshold - knee`` and
    ``hi = threshold + knee``:

    * ``v <= lo`` maps to ``0.0``;
    * ``v >= hi`` maps to ``v - threshold``;
    * between them (when ``knee > 0``), with
      ``u = (v - lo) / (2 * knee)``, map to ``knee * u * u``.

    With ``knee == 0`` this reduces to ``0.0`` for ``v <= threshold`` and
    ``v - threshold`` above it. The result is continuous, non-negative, has
    the same shape as ``img``, and carries values linearly in ``[0.0, 1.0]``.
    This is a high-pass bright-pass map, not a binarisation. Results must match
    the piecewise definition within ``1e-6``; this is intentionally a soft
    threshold so bright edges do not become blocky.
    """
    _validate_image(img)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
            or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise BloomError("threshold must be a finite value in [0.0, 1.0]")
    if not isinstance(knee, (int, float)) or isinstance(knee, bool) \
            or not math.isfinite(knee) or not 0.0 <= knee <= threshold:
        raise BloomError("knee must satisfy 0.0 <= knee <= threshold")

    threshold = float(threshold)
    knee = float(knee)
    lo = threshold - knee
    hi = threshold + knee

    out = []
    for row in img:
        out_row = []
        for v in row:
            if knee == 0.0:
                out_row.append(0.0 if v <= threshold else v - threshold)
            elif v <= lo:
                out_row.append(0.0)
            elif v >= hi:
                out_row.append(v - threshold)
            else:
                u = (v - lo) / (2.0 * knee)
                out_row.append(knee * u * u)
        out.append(out_row)
    return out


def downsample(img: list[list[float]]) -> list[list[float]]:
    """Downsample a linear-light image by one 2x2 mean-pooling level.

    ``img`` must be a non-empty rectangular image with finite values in
    ``[0.0, 1.0]``. The output shape is ``(ceil(height / 2), ceil(width / 2))``.
    Output pixel ``(y, x)`` is the arithmetic mean of every in-bounds source
    pixel in rows ``2*y`` through ``2*y + 1`` and columns ``2*x`` through
    ``2*x + 1``. At an odd edge the window is clipped to existing pixels, so
    the divisor is the actual sample count (1, 2, or 4) and no source pixel,
    including the final column or row, is discarded. Pooling is linear and the
    result must match within ``1e-6``.
    """
    height, width = _validate_image(img)
    out_h = (height + 1) // 2
    out_w = (width + 1) // 2

    out = []
    for y in range(out_h):
        out_row = []
        for x in range(out_w):
            total = 0.0
            count = 0
            for sy in (2 * y, 2 * y + 1):
                if sy >= height:
                    continue
                for sx in (2 * x, 2 * x + 1):
                    if sx >= width:
                        continue
                    total += img[sy][sx]
                    count += 1
            out_row.append(total / count)
        out.append(out_row)
    return out


def upsample(img: list[list[float]], size: tuple[int, int]) -> list[list[float]]:
    """Bilinearly resize a linear-light image to ``size == (height, width)``.

    ``img`` must be a non-empty rectangular image with finite values in
    ``[0.0, 1.0]``; both target dimensions must be positive integers. The
    output has exactly the requested size and is sampled with corner-aligned
    coordinates: for target index ``i`` the source coordinate is
    ``i * (source_extent - 1) / (target_extent - 1)`` when the target extent is
    greater than 1, otherwise ``0.0``. Coordinates are clamped to the source
    bounds, and the four surrounding samples are clamped at the borders before
    bilinear weighting. Thus resizing to the source size is an exact copy.

    This is genuinely bilinear interpolation, not nearest-neighbour: fractional
    coordinates blend the adjacent samples. Interpolation is linear and the
    result must match within ``1e-6``.
    """
    src_h, src_w = _validate_image(img)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise BloomError("size must be a (height, width) pair")
    dst_h, dst_w = size
    for dim in (dst_h, dst_w):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise BloomError("target dimensions must be positive integers")

    def source_coord(i, src_extent, dst_extent):
        if dst_extent > 1:
            coord = i * (src_extent - 1) / (dst_extent - 1)
        else:
            coord = 0.0
        return min(max(coord, 0.0), src_extent - 1)

    out = []
    for y in range(dst_h):
        sy = source_coord(y, src_h, dst_h)
        y0 = int(math.floor(sy))
        y1 = min(y0 + 1, src_h - 1)
        fy = sy - y0
        out_row = []
        for x in range(dst_w):
            sx = source_coord(x, src_w, dst_w)
            x0 = int(math.floor(sx))
            x1 = min(x0 + 1, src_w - 1)
            fx = sx - x0
            top = img[y0][x0] * (1.0 - fx) + img[y0][x1] * fx
            bottom = img[y1][x0] * (1.0 - fx) + img[y1][x1] * fx
            out_row.append(top * (1.0 - fy) + bottom * fy)
        out.append(out_row)
    return out


def _max_levels(height, width):
    """Depth available while both dimensions strictly shrink by ceil-half."""
    levels = 1
    while height > 1 and width > 1:
        height = (height + 1) // 2
        width = (width + 1) // 2
        levels += 1
    return levels


def build_pyramid(img: list[list[float]], levels: int) -> list:
    """Build a strictly shrinking linear-light pyramid.

    ``img`` must be a non-empty rectangular image with finite values in
    ``[0.0, 1.0]``. ``levels`` counts total layers, including the original:
    ``levels == 1`` returns one layer whose dimensions equal ``img``; each later
    layer is ``downsample(previous_layer)``. Layer 0 must be an independent
    copy, not an alias of ``img``.

    Every step must strictly reduce both height and width, so the layer count
    cannot exceed the depth available before the smaller dimension reaches 1.
    Compute that limit iteratively with the same ceil-half rule as
    :func:`downsample`. ``levels`` must be an integer in ``[1, limit]``;
    ``levels == 0``, negative values, and too many levels raise
    :class:`BloomError` instead of dividing by zero or emitting duplicate 1x1
    layers. The result is a list of images with strictly decreasing dimensions,
    and values must match repeated downsampling within ``1e-6``.
    """
    height, width = _validate_image(img)
    if not isinstance(levels, int) or isinstance(levels, bool):
        raise BloomError("levels must be an integer")
    limit = _max_levels(height, width)
    if levels < 1:
        raise BloomError("levels must be at least 1")
    if levels > limit:
        raise BloomError(
            f"levels={levels} exceeds the strictly shrinking depth {limit} "
            f"for a {height}x{width} image"
        )

    pyramid = [_copy_image(img)]
    for _ in range(levels - 1):
        pyramid.append(downsample(pyramid[-1]))
    return pyramid


def combine(pyramid: list[list[list[float]]], weights: list[float]) -> list[list[float]]:
    """Combine pyramid layers into one image at layer 0's resolution.

    ``pyramid`` must be a non-empty list of rectangular non-empty layers with
    finite values in ``[0.0, 1.0]`` (layers generated by
    :func:`build_pyramid` have decreasing sizes). ``weights`` must contain one
    finite, non-negative weight per layer and have a positive total. Otherwise
    raise :class:`BloomError`.

    Each layer is first resampled to the dimensions of layer 0 using the
    bilinear, edge-clamped rule of :func:`upsample`. Weights are then normalised
    by their sum, so the result is the convex weighted mean of the resampled
    layers and cannot become brighter merely because there are many layers.
    No two differently shaped arrays are added directly. The returned image has
    exactly layer 0's dimensions and values in ``[0.0, 1.0]``; results must
    match within ``1e-6``.
    """
    if not isinstance(pyramid, (list, tuple)) or len(pyramid) == 0:
        raise BloomError("pyramid must be a non-empty list of layers")
    for layer in pyramid:
        _validate_image(layer, name="pyramid layer")
    if not isinstance(weights, (list, tuple)) or len(weights) != len(pyramid):
        raise BloomError("weights must contain one value per layer")
    for w in weights:
        if not isinstance(w, (int, float)) or isinstance(w, bool) \
                or not math.isfinite(w) or w < 0.0:
            raise BloomError("weights must be finite and non-negative")
    total = float(sum(weights))
    if total <= 0.0:
        raise BloomError("weights must have a positive total")

    base_h, base_w = len(pyramid[0]), len(pyramid[0][0])
    resampled = [
        layer if (len(layer), len(layer[0])) == (base_h, base_w)
        else upsample(layer, (base_h, base_w))
        for layer in pyramid
    ]

    out = []
    for y in range(base_h):
        out_row = []
        for x in range(base_w):
            acc = 0.0
            for layer, w in zip(resampled, weights):
                acc += w * layer[y][x]
            out_row.append(acc / total)
        out.append(out_row)
    return out


def apply_bloom(
    img: list[list[float]],
    threshold: float,
    levels: int,
    strength: float,
) -> list[list[float]]:
    """Run the full linear-light bloom chain and return a new image.

    ``img`` must be a non-empty rectangular image with finite values in
    ``[0.0, 1.0]``. ``threshold`` must be in ``[0.0, 1.0]`` and ``levels`` must
    be valid for the image under :func:`build_pyramid`; invalid values raise
    :class:`BloomError`. ``strength`` is clamped to ``[0.0, 1.0]`` (out-of-range
    values are not allowed to overdrive highlights).

    The chain is: build the bright-pass map with
    ``threshold(img, threshold, 0.1 * threshold)``; build a pyramid of that map
    with ``levels`` layers; combine all layers with equal weights (the
    normalisation in :func:`combine` makes this an average); then add the
    resulting bloom map to ``img`` as ``img + strength * bloom``, clipping each
    final pixel to ``[0.0, 1.0]``. If ``strength == 0.0``, return an independent
    copy of ``img``. If every input pixel is exactly ``0.0``, short-circuit and
    return an all-``0.0`` image: black must stay exactly black with no
    accumulated floating-point haze.

    Operations are linear-light. Non-exact values must match within ``1e-6``;
    the all-black and strength-zero identity cases are the explicit exceptions.
    """
    _validate_image(img)
    if not isinstance(strength, (int, float)) or isinstance(strength, bool) \
            or not math.isfinite(strength):
        raise BloomError("strength must be a finite number")
    strength = min(max(float(strength), 0.0), 1.0)

    if strength == 0.0:
        return _copy_image(img)
    if all(value == 0.0 for row in img for value in row):
        return _copy_image(img)

    bright = globals()["threshold"](img, threshold, 0.1 * threshold)
    pyramid = build_pyramid(bright, levels)
    bloom = combine(pyramid, [1.0] * len(pyramid))

    out = []
    for img_row, bloom_row in zip(img, bloom):
        out_row = []
        for v, b in zip(img_row, bloom_row):
            out_row.append(min(max(v + strength * b, 0.0), 1.0))
        out.append(out_row)
    return out


def validate_pyramid(pyramid: list[list[list[float]]]) -> list[str]:
    """Return human-readable structural problems with ``pyramid``.

    The returned list is empty exactly when the structure is valid. This
    function reports issues rather than raising for bad pyramid data. Inspect:

    * an empty pyramid (no levels);
    * an empty, ragged, or non-rectangular layer;
    * dimensions that are not strictly decreasing between consecutive layers;
    * a layer count exceeding the strictly shrinking depth available from
      layer 0 (a level-count mismatch for that geometry).

    Sizes are integer structural properties and are checked exactly. Numeric
    bloom contracts elsewhere use an absolute tolerance of ``1e-6``.
    """
    issues = []
    if not isinstance(pyramid, (list, tuple)) or len(pyramid) == 0:
        return ["pyramid is empty: no levels"]

    shapes = []
    for index, layer in enumerate(pyramid):
        if not isinstance(layer, (list, tuple)) or len(layer) == 0:
            issues.append(f"layer {index} is empty")
            shapes.append(None)
            continue
        width = len(layer[0]) if isinstance(layer[0], (list, tuple)) else 0
        if width == 0:
            issues.append(f"layer {index} has empty rows")
            shapes.append(None)
            continue
        if any(not isinstance(row, (list, tuple)) or len(row) != width for row in layer):
            issues.append(f"layer {index} is ragged (non-rectangular)")
            shapes.append(None)
            continue
        shapes.append((len(layer), width))

    known = [shape for shape in shapes if shape is not None]
    for index in range(1, len(shapes)):
        prev, cur = shapes[index - 1], shapes[index]
        if prev is None or cur is None:
            continue
        if cur[0] >= prev[0] or cur[1] >= prev[1]:
            issues.append(
                f"layers {index - 1} and {index} are not strictly decreasing: "
                f"{prev[0]}x{prev[1]} -> {cur[0]}x{cur[1]}"
            )

    if shapes[0] is not None:
        limit = _max_levels(*shapes[0])
        if len(pyramid) > limit:
            issues.append(
                f"pyramid has {len(pyramid)} levels but layer 0 geometry "
                f"{shapes[0][0]}x{shapes[0][1]} supports at most {limit}"
            )
    return issues


def bloom_batch(images: list[list[list[float]]], config: dict) -> dict:
    """Process a batch of images atomically and independently.

    ``images`` is a non-empty list of linear-light images and ``config`` is a
    mapping with exactly the keys ``threshold``, ``levels``, and ``strength``
    accepted by :func:`apply_bloom`. Return a dictionary with:

    * ``"ok"``: ``True`` only when every image succeeded;
    * ``"results"``: one bloomed image per input, in the original order, on
      success; ``[]`` on failure;
    * ``"errors"``: a list of strings, empty on success; on failure it
      identifies the offending image index or config problem.

    Every image must use a freshly built pyramid; no pyramid, buffer, or mutable
    accumulator may be shared between images. Validate the config and all images
    before producing output. If any image is malformed or any call raises
    :class:`BloomError`, roll the whole batch back and return ``ok == False``
    with no partial results. Do not raise for per-image data errors. Results
    must match individual :func:`apply_bloom` calls within ``1e-6``.
    """
    failure = {"ok": False, "results": [], "errors": []}

    if not isinstance(images, (list, tuple)) or len(images) == 0:
        failure["errors"].append("images must be a non-empty list")
        return failure

    expected_keys = {"threshold", "levels", "strength"}
    if not isinstance(config, dict):
        failure["errors"].append("config must be a mapping")
        return failure
    if set(config.keys()) != expected_keys:
        failure["errors"].append(
            f"config must have exactly the keys {sorted(expected_keys)}"
        )
        return failure

    # Validate every image before producing any output.
    for index, img in enumerate(images):
        try:
            _validate_image(img, name=f"images[{index}]")
        except BloomError as exc:
            failure["errors"].append(f"image {index}: {exc}")
    if failure["errors"]:
        return failure

    results = []
    for index, img in enumerate(images):
        try:
            results.append(apply_bloom(
                img,
                config["threshold"],
                config["levels"],
                config["strength"],
            ))
        except BloomError as exc:
            failure["errors"].append(f"image {index}: {exc}")
            return failure
        except (TypeError, ValueError) as exc:
            failure["errors"].append(f"image {index}: {exc}")
            return failure

    return {"ok": True, "results": results, "errors": []}
