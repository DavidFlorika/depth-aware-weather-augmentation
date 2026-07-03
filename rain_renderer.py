import os
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


OUTPUT_DIR = Path(os.environ.get("RAIN_PROJECT_OUTPUT_DIR", "/content/rain_project_outputs"))
try:
    SAMPLE_DIR = OUTPUT_DIR / "generated_samples"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    OUTPUT_DIR = Path.cwd() / "rain_project_outputs"
    SAMPLE_DIR = OUTPUT_DIR / "generated_samples"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def _as_uint8_bgr(image):
    if image is None:
        raise ValueError("Input image is None. Check the file path.")

    img = np.asarray(image)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")

    if img.dtype == np.uint8:
        return img.copy()

    img = np.nan_to_num(img)

    if img.max() <= 1.0:
        img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def _normalize01(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)

    lo, hi = np.percentile(x, [1, 99])
    x = np.clip(x, lo, hi)

    return (x - x.min()) / (x.max() - x.min() + eps)


def normalize_depth_for_rain(depth_map, image_shape=None, close_is_high=True):
    """
    Convert raw model depth into distance01:
        0.0 = close to camera
        1.0 = far from camera
    """
    d = _normalize01(depth_map)

    if close_is_high:
        distance01 = 1.0 - d
    else:
        distance01 = d

    if image_shape is not None:
        h, w = image_shape[:2]

        if distance01.shape[:2] != (h, w):
            distance01 = cv2.resize(distance01, (w, h), interpolation=cv2.INTER_LINEAR)

    return np.clip(distance01.astype(np.float32), 0.0, 1.0)


def estimate_depth(image_bgr, depth_model=None):
    image_bgr = _as_uint8_bgr(image_bgr)

    if depth_model is None:
        depth_model = globals().get("model")

    if depth_model is None:
        raise RuntimeError("Depth model is not available. Pass depth_model or load `model` first.")

    depth = depth_model.infer_image(image_bgr)
    return np.asarray(depth, dtype=np.float32)


def _severity_value(severity):
    if isinstance(severity, str):
        table = {
            "light": 0.25,
            "medium": 0.55,
            "heavy": 0.85,
        }

        if severity.lower() not in table:
            raise ValueError("severity must be 'light', 'medium', 'heavy', or a number.")

        return table[severity.lower()]

    severity = float(severity)

    if severity > 1.0:
        severity = severity / 5.0

    return float(np.clip(severity, 0.05, 1.0))


def _rain_visibility_severity(severity_value):
    """
    Small visibility lift for light/medium rain streaks.

    Heavy rain is already visible, so the boost smoothly fades out before the
    heavy range. Fog/veil code should keep using the original severity.
    """
    s = float(np.clip(severity_value, 0.05, 1.0))
    boost = 0.10 * (1.0 - float(_smoothstep(0.58, 0.84, s)))
    boost += 0.05 * (1.0 - float(_smoothstep(0.25, 0.55, s)))
    return float(np.clip(s + boost, 0.05, 1.0))


def _smoothstep(edge0, edge1, x):
    x = np.asarray(x, dtype=np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _srgb_to_linear01(x):
    x = np.asarray(x, dtype=np.float32) / 255.0
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear01_to_srgb255(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0.0, 255.0)


def _linear_luminance01(image_bgr):
    lin = _srgb_to_linear01(image_bgr)
    return np.clip(
        0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2],
        0.0,
        1.0,
    ).astype(np.float32)


def _depth_edge_mask(distance01):
    d = np.asarray(distance01, dtype=np.float32)
    d = cv2.GaussianBlur(d, (0, 0), 0.8)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    scale = np.percentile(edge, 96) + 1e-6
    edge = np.clip(edge / scale, 0.0, 1.0)
    return cv2.GaussianBlur(edge, (0, 0), 0.7).astype(np.float32)


def _visibility_for_rain_layer(distance01, layer_depth, softness=0.035, edge_mask=None):
    """
    Rain at layer_depth is visible only where it is in front of the scene.

    distance01 is the visible scene surface distance. A rain particle with
    smaller distance is closer to the camera, so it should be hidden where
    scene distance is smaller than the particle distance.
    """
    scene_distance = np.asarray(distance01, dtype=np.float32)
    visibility = _smoothstep(-softness, softness, scene_distance - float(layer_depth))

    if edge_mask is not None:
        boundary = np.exp(-np.square((scene_distance - float(layer_depth)) / (softness + 1e-6)))
        visibility *= 1.0 - 0.45 * np.clip(edge_mask, 0.0, 1.0) * boundary

    return np.clip(visibility, 0.0, 1.0).astype(np.float32)


def _rain_density_from_depth(distance01):
    """
    Legacy density helper retained for compatibility.

    Farther visible surfaces can reveal more rain volume, but the curve is
    sublinear near the camera so foreground objects are not overpainted.
    """
    d = np.clip(np.asarray(distance01, dtype=np.float32), 0.0, 1.0)
    density = 0.06 + 0.82 * np.power(d, 1.35)
    return np.clip(density, 0.0, 1.0)


def _draw_rain_streaks(
    shape,
    rng,
    n_streaks,
    length_px,
    angle_deg=105,
    thickness=1,
    intensity_range=(150, 255),
    acceptance_map=None,
):
    """
    Baseline non-depth-aware rain helper.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    angle = np.deg2rad(angle_deg)

    attempts = int(n_streaks * (2.2 if acceptance_map is not None else 1.0)) + 50
    xs = rng.integers(0, w, size=attempts)
    ys = rng.integers(0, h, size=attempts)

    accepted = []

    for x, y in zip(xs, ys):
        if acceptance_map is not None and rng.random() > acceptance_map[y, x]:
            continue

        accepted.append((x, y))

        if len(accepted) >= n_streaks:
            break

    for x, y in accepted:
        length_jitter = rng.uniform(0.75, 1.25)
        local_len = max(3, int(length_px * length_jitter))

        angle_jitter = np.deg2rad(rng.normal(0, 3.0))
        dx = np.cos(angle + angle_jitter)
        dy = np.sin(angle + angle_jitter)

        x2 = int(round(x + dx * local_len))
        y2 = int(round(y + dy * local_len))

        intensity = float(rng.uniform(*intensity_range))

        cv2.line(
            mask,
            (int(x), int(y)),
            (int(x2), int(y2)),
            intensity,
            int(thickness),
            lineType=cv2.LINE_AA,
        )

    return np.clip(mask / 255.0, 0.0, 1.0)


def _rain_particle_depth_parameters(distance_value, base_length_px, base_thickness, severity_value, rng=None):
    """
    Legacy per-particle parameter helper retained for notebooks that call it.

    The current depth-aware renderer uses layered volumetric passes instead.
    """
    d = float(np.clip(distance_value, 0.0, 1.0))

    length_scale = 1.18 - 0.58 * d
    length_px = max(3, int(round(base_length_px * length_scale)))

    thickness = max(1, int(base_thickness))
    if severity_value >= 0.80 and d < 0.22:
        thickness += 1

    brightness = 215.0 - 25.0 * d
    opacity = 0.025 + 0.075 * severity_value
    opacity *= 1.12 - 0.55 * d

    if rng is not None:
        length_px = max(3, int(round(length_px * rng.uniform(0.85, 1.20))))
        brightness = brightness + rng.normal(0.0, 7.0)
        opacity = opacity * rng.uniform(0.85, 1.20)

    brightness = float(np.clip(brightness, 135.0, 245.0))
    opacity = float(np.clip(opacity, 0.012, 0.18))

    return length_px, thickness, brightness, opacity


def _draw_depth_scaled_rain_streaks(
    shape,
    rng,
    n_streaks,
    base_length_px,
    severity_value,
    angle_deg=105,
    base_thickness=1,
    distance01=None,
    acceptance_map=None,
):
    """
    Legacy depth-aware streak helper retained for compatibility.

    New depth-aware rendering is handled by _render_layered_depth_rain.
    """
    h, w = shape[:2]

    if distance01 is None:
        distance01 = np.full((h, w), 0.5, dtype=np.float32)
    else:
        distance01 = np.asarray(distance01, dtype=np.float32)

        if distance01.shape[:2] != (h, w):
            distance01 = cv2.resize(distance01, (w, h), interpolation=cv2.INTER_LINEAR)

        distance01 = np.clip(distance01, 0.0, 1.0)

    if acceptance_map is not None:
        acceptance_map = np.asarray(acceptance_map, dtype=np.float32)

        if acceptance_map.shape[:2] != (h, w):
            acceptance_map = cv2.resize(acceptance_map, (w, h), interpolation=cv2.INTER_LINEAR)

        acceptance_map = np.clip(acceptance_map, 0.0, 1.0)

    alpha_mask = np.zeros((h, w), dtype=np.float32)
    brightness_map = np.zeros((h, w), dtype=np.float32)

    angle = np.deg2rad(angle_deg)

    attempts = int(n_streaks * (4.2 if acceptance_map is not None else 1.5)) + 250

    xs = rng.integers(0, w, size=attempts)
    ys = rng.integers(0, h, size=attempts)

    drawn = 0

    for x, y in zip(xs, ys):
        if acceptance_map is not None and rng.random() > acceptance_map[y, x]:
            continue

        local_distance = float(distance01[y, x])

        local_len, local_thickness, brightness, opacity = _rain_particle_depth_parameters(
            local_distance,
            base_length_px=base_length_px,
            base_thickness=base_thickness,
            severity_value=severity_value,
            rng=rng,
        )

        angle_jitter = np.deg2rad(rng.normal(0, 2.5))
        local_angle = angle + angle_jitter

        dx = np.cos(local_angle)
        dy = np.sin(local_angle)

        half_len = max(2.0, local_len / 2.0)

        x1 = int(round(x - dx * half_len))
        y1 = int(round(y - dy * half_len))
        x2 = int(round(x + dx * half_len))
        y2 = int(round(y + dy * half_len))

        cv2.line(
            alpha_mask,
            (x1, y1),
            (x2, y2),
            float(opacity),
            int(local_thickness),
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            brightness_map,
            (x1, y1),
            (x2, y2),
            float(brightness),
            int(local_thickness),
            lineType=cv2.LINE_AA,
        )

        drawn += 1

        if drawn >= n_streaks:
            break

    brightness_map = np.where(alpha_mask > 0.0, brightness_map, 220.0)

    return np.clip(alpha_mask, 0.0, 1.0), np.clip(brightness_map, 120.0, 250.0)


def _rain_fog_darkening_map(shape, rng, severity, distance01=None, strength_range=None):
    """
    Legacy subtractive rain-fog mask used by the simple 2D baseline.
    """
    h, w = shape[:2]
    s = _severity_value(severity)

    if strength_range is None:
        low = 5.0 + 15.0 * s
        high = 12.0 + 26.0 * s
    else:
        low, high = strength_range
        low = float(low)
        high = float(high)

        if high < low:
            raise ValueError("strength_range must be ordered as (min_value, max_value).")

    base_drop = float(rng.uniform(low, high))
    darken = np.full((h, w), base_drop, dtype=np.float32)

    if distance01 is not None:
        d = np.asarray(distance01, dtype=np.float32)

        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

        d = np.clip(d, 0.0, 1.0)

        darken += base_drop * (0.08 + 0.12 * s) * d

    if h > 1 and w > 1:
        noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        sigma = max(4.0, min(h, w) * 0.03)
        noise = cv2.GaussianBlur(noise, (0, 0), sigma)
        noise = noise / (np.max(np.abs(noise)) + 1e-6)
        darken *= 1.0 + 0.06 * noise

    return np.clip(darken, 0.0, 50.0)


def _composite_rain(
    image_bgr,
    rain_mask,
    alpha,
    veil_map=None,
    rain_brightness=None,
    darken_map=None,
    min_rain_delta=0.0,
):
    """
    Composite rain over an image in linear light.

    This keeps the old function signature for the simple baseline while
    avoiding gamma-space blending artifacts.
    """
    img = image_bgr.astype(np.float32)

    if darken_map is not None:
        dark = np.asarray(darken_map, dtype=np.float32)

        if dark.ndim == 2:
            dark = dark[..., None]

        dark = np.clip(dark, 0.0, 255.0)
        img = np.maximum(img - dark, 0.0)

    rain_mask = np.clip(rain_mask.astype(np.float32), 0.0, 1.0)

    if rain_brightness is None:
        rain_rgb = np.full_like(img, 225.0, dtype=np.float32)
    else:
        brightness = np.clip(np.asarray(rain_brightness, dtype=np.float32), 0.0, 255.0)

        if brightness.ndim == 2:
            rain_rgb = np.repeat(brightness[..., None], 3, axis=2)
        else:
            rain_rgb = brightness

    a = np.clip(alpha * rain_mask, 0.0, 1.0)[..., None]
    base_lin = _srgb_to_linear01(img)
    rain_lin = _srgb_to_linear01(rain_rgb)
    blended_lin = base_lin * (1.0 - a) + rain_lin * a
    blended = _linear01_to_srgb255(blended_lin)

    if min_rain_delta > 0:
        rain_exists = rain_mask[..., None] > 1e-6
        delta = blended - img

        direction = np.sign(rain_rgb - img)
        direction = np.where(direction == 0.0, 1.0, direction)

        small_delta = np.abs(delta) < min_rain_delta
        forced = img + direction * min_rain_delta

        blended = np.where(rain_exists & small_delta, forced, blended)

    out = blended

    if veil_map is not None:
        veil = np.clip(veil_map.astype(np.float32), 0.0, 1.0)[..., None]
        veil_color = np.full_like(img, 210.0, dtype=np.float32)
        out_lin = _srgb_to_linear01(out)
        veil_lin = _srgb_to_linear01(veil_color)
        out = _linear01_to_srgb255(out_lin * (1.0 - veil) + veil_lin * veil)

    return np.clip(out, 0, 255).astype(np.uint8)


def _make_motion_streak_kernel(length_px, angle_deg, width_px=1.0, taper_power=1.45):
    length_px = max(3.0, float(length_px))
    width_px = max(0.45, float(width_px))
    half = length_px * 0.5
    radius = int(np.ceil(half + width_px * 4.0 + 3.0))
    radius = max(radius, 3)

    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    theta = np.deg2rad(float(angle_deg))
    ct = np.cos(theta)
    st = np.sin(theta)

    along = xx * ct + yy * st
    across = -xx * st + yy * ct

    along_weight = np.clip(1.0 - np.abs(along) / (half + 1e-6), 0.0, 1.0) ** taper_power
    across_sigma = max(0.35, width_px * 0.55)
    across_weight = np.exp(-0.5 * np.square(across / across_sigma))

    support = (np.abs(along) <= half) & (np.abs(across) <= width_px * 3.0 + 1.0)
    kernel = along_weight * across_weight * support.astype(np.float32)
    peak = float(kernel.max())

    if peak <= 1e-8:
        kernel[radius, radius] = 1.0
    else:
        kernel = kernel / peak

    return kernel.astype(np.float32)


def _rain_layer_specs(image_shape, severity_value):
    h, w = image_shape[:2]
    m = float(min(h, w))
    area_scale = h * w / 1000.0
    s = float(severity_value)

    return [
        {
            "name": "near",
            "depth": 0.08,
            "count": int(area_scale * (0.025 + 0.13 * s)),
            "length": m * (0.045 + 0.080 * s),
            "width": 1.75 + 2.20 * s,
            "opacity": 0.020 + 0.055 * s,
            "delta": 70.0 + 60.0 * s,
            "shadow": 8.0 + 12.0 * s,
            "softness": 0.060,
            "passes": 2,
            "defocus": 0.65 + 1.25 * s,
            "angle_jitter": 6.0,
        },
        {
            "name": "mid",
            "depth": 0.30,
            "count": int(area_scale * (0.45 + 2.55 * s)),
            "length": m * (0.018 + 0.038 * s),
            "width": 0.85 + 0.85 * s,
            "opacity": 0.018 + 0.052 * s,
            "delta": 52.0 + 54.0 * s,
            "shadow": 5.0 + 10.0 * s,
            "softness": 0.050,
            "passes": 3,
            "defocus": 0.15 + 0.35 * s,
            "angle_jitter": 4.0,
        },
        {
            "name": "far",
            "depth": 0.62,
            "count": int(area_scale * (1.20 + 5.20 * s)),
            "length": m * (0.007 + 0.020 * s),
            "width": 0.55 + 0.38 * s,
            "opacity": 0.007 + 0.026 * s,
            "delta": 32.0 + 34.0 * s,
            "shadow": 2.0 + 7.0 * s,
            "softness": 0.070,
            "passes": 3,
            "defocus": 0.10 + 0.25 * s,
            "angle_jitter": 3.0,
        },
        {
            "name": "background",
            "depth": 0.84,
            "count": int(area_scale * (1.65 + 6.50 * s)),
            "length": m * (0.004 + 0.012 * s),
            "width": 0.45 + 0.25 * s,
            "opacity": 0.004 + 0.014 * s,
            "delta": 20.0 + 28.0 * s,
            "shadow": 1.0 + 4.0 * s,
            "softness": 0.090,
            "passes": 2,
            "defocus": 0.35 + 0.45 * s,
            "angle_jitter": 2.5,
        },
    ]


def _accepted_random_points(rng, h, w, n_points, acceptance_map):
    n_points = int(max(0, n_points))

    if n_points == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    acceptance = np.clip(np.asarray(acceptance_map, dtype=np.float32), 0.0, 1.0)
    mean_acceptance = float(acceptance.mean())

    if mean_acceptance <= 1e-5:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    xs_parts = []
    ys_parts = []
    collected = 0

    for _ in range(5):
        expected = int(n_points / (mean_acceptance + 1e-4) * 1.25)
        attempts = min(max(expected, n_points * 2 + 256), n_points * 12 + 2048)
        xs = rng.integers(0, w, size=attempts, dtype=np.int32)
        ys = rng.integers(0, h, size=attempts, dtype=np.int32)
        keep = rng.random(attempts) < acceptance[ys, xs]

        if np.any(keep):
            kept_xs = xs[keep]
            kept_ys = ys[keep]
            xs_parts.append(kept_xs)
            ys_parts.append(kept_ys)
            collected += kept_xs.size

        if collected >= n_points:
            break

    if collected == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    xs_all = np.concatenate(xs_parts)[:n_points]
    ys_all = np.concatenate(ys_parts)[:n_points]
    return xs_all.astype(np.int32), ys_all.astype(np.int32)


def _render_motion_layer(image_bgr, rng, spec, visibility, luma, angle_deg):
    h, w = image_bgr.shape[:2]
    passes = max(1, int(spec["passes"]))
    total_tau = np.zeros((h, w), dtype=np.float32)
    total_delta_tau = np.zeros((h, w), dtype=np.float32)

    count = max(0, int(spec["count"]))

    for pass_idx in range(passes):
        pass_count = count // passes
        if pass_idx < count % passes:
            pass_count += 1

        if pass_count <= 0:
            continue

        local_angle = float(angle_deg + rng.normal(0.0, spec["angle_jitter"]))
        local_length = max(3.0, float(spec["length"]) * rng.uniform(0.84, 1.20))
        local_width = max(0.45, float(spec["width"]) * rng.uniform(0.86, 1.16))
        kernel = _make_motion_streak_kernel(local_length, local_angle, local_width)

        xs, ys = _accepted_random_points(rng, h, w, pass_count, visibility)
        if xs.size == 0:
            continue

        seed_tau = np.zeros((h, w), dtype=np.float32)
        seed_delta_tau = np.zeros((h, w), dtype=np.float32)

        seed_luma = np.clip(luma[ys, xs], 0.0, 1.0)
        contrast = 0.34 + 1.05 * np.power(1.0 - seed_luma, 0.72)
        deltas = float(spec["delta"]) * contrast - float(spec["shadow"]) * np.power(seed_luma, 1.15)
        deltas += rng.normal(0.0, 4.0 + 5.0 * float(spec["opacity"]), size=xs.size)
        deltas = np.clip(deltas, -16.0, 135.0).astype(np.float32)

        opacities = float(spec["opacity"]) * rng.lognormal(mean=0.0, sigma=0.22, size=xs.size)
        opacities = np.clip(opacities, 0.001, 0.18).astype(np.float32)

        np.add.at(seed_tau, (ys, xs), opacities)
        np.add.at(seed_delta_tau, (ys, xs), opacities * deltas)

        layer_tau = cv2.filter2D(seed_tau, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
        layer_delta_tau = cv2.filter2D(seed_delta_tau, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)

        layer_tau *= visibility
        layer_delta_tau *= visibility

        defocus = float(spec["defocus"])
        if defocus > 0.05:
            layer_tau = cv2.GaussianBlur(layer_tau, (0, 0), defocus)
            layer_delta_tau = cv2.GaussianBlur(layer_delta_tau, (0, 0), defocus)

        total_tau += layer_tau
        total_delta_tau += layer_delta_tau

    return total_tau, total_delta_tau


def _render_lens_droplets(shape, rng, severity_value, luma):
    h, w = shape[:2]
    s = float(severity_value)
    tau = np.zeros((h, w), dtype=np.float32)
    delta_tau = np.zeros((h, w), dtype=np.float32)

    if s < 0.45:
        return tau, delta_tau

    min_dim = min(h, w)
    count = int((h * w / 320000.0) * (1.0 + 6.0 * s))

    for _ in range(count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        rx = max(3, int(min_dim * rng.uniform(0.006, 0.024)))
        ry = max(5, int(rx * rng.uniform(1.4, 3.2)))
        angle = float(rng.normal(8.0, 12.0))
        opacity = float(rng.uniform(0.006, 0.020) * (0.5 + s))

        cv2.ellipse(tau, (x, y), (rx, ry), angle, 0, 360, opacity, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(tau, (x, y), (rx, ry), angle, 0, 360, opacity * 2.1, 1, lineType=cv2.LINE_AA)

    if count > 0:
        tau = cv2.GaussianBlur(tau, (0, 0), 1.2 + 1.8 * s)
        local_delta = 24.0 + 85.0 * np.power(1.0 - np.clip(luma, 0.0, 1.0), 0.8)
        delta_tau = tau * local_delta.astype(np.float32)

    return tau, delta_tau


def _add_fine_mist_spray(total_tau, total_delta_tau, distance01, edge_mask, rng, severity_value):
    h, w = distance01.shape[:2]
    s = float(severity_value)

    if s < 0.20:
        return total_tau, total_delta_tau

    far_visibility = _visibility_for_rain_layer(distance01, 0.74, softness=0.12, edge_mask=edge_mask)
    grain = rng.random((h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.45)
    grain = np.clip((grain - 0.42) / 0.58, 0.0, 1.0)

    mist_tau = (0.0015 + 0.009 * s) * far_visibility * (0.25 + 0.75 * distance01) * grain
    mist_delta = 10.0 + 28.0 * s

    total_tau += mist_tau
    total_delta_tau += mist_tau * mist_delta
    return total_tau, total_delta_tau


def _render_layered_depth_rain(image_bgr, distance01, rng, severity_value, angle_deg=105):
    h, w = image_bgr.shape[:2]
    d = np.asarray(distance01, dtype=np.float32)

    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

    d = np.clip(d, 0.0, 1.0)
    edge_mask = _depth_edge_mask(d)
    luma = _linear_luminance01(image_bgr)

    total_tau = np.zeros((h, w), dtype=np.float32)
    total_delta_tau = np.zeros((h, w), dtype=np.float32)

    for spec in _rain_layer_specs(image_bgr.shape, severity_value):
        visibility = _visibility_for_rain_layer(
            d,
            spec["depth"],
            softness=spec["softness"],
            edge_mask=edge_mask,
        )
        layer_tau, layer_delta_tau = _render_motion_layer(
            image_bgr,
            rng,
            spec,
            visibility,
            luma,
            angle_deg,
        )
        total_tau += layer_tau
        total_delta_tau += layer_delta_tau

    total_tau, total_delta_tau = _add_fine_mist_spray(
        total_tau,
        total_delta_tau,
        d,
        edge_mask,
        rng,
        severity_value,
    )

    lens_tau, lens_delta_tau = _render_lens_droplets(image_bgr.shape, rng, severity_value, luma)
    total_tau += lens_tau
    total_delta_tau += lens_delta_tau

    alpha = 1.0 - np.exp(-np.clip(total_tau, 0.0, 3.5))
    delta = total_delta_tau / (total_tau + 1e-6)
    delta = np.where(total_tau > 1e-7, delta, 0.0)

    return np.clip(alpha, 0.0, 0.62).astype(np.float32), np.clip(delta, -18.0, 145.0).astype(np.float32)


def _estimate_haze_airlight_bgr(image_bgr):
    img = _as_uint8_bgr(image_bgr).astype(np.float32)
    luma = _linear_luminance01(img)
    threshold = np.percentile(luma, 88)
    bright = img[luma >= threshold]

    if bright.size == 0:
        airlight = np.percentile(img.reshape(-1, 3), 82, axis=0)
    else:
        airlight = np.percentile(bright.reshape(-1, 3), 70, axis=0)

    cool_overcast = np.array([188.0, 192.0, 194.0], dtype=np.float32)
    airlight = 0.68 * airlight.astype(np.float32) + 0.32 * cool_overcast
    return np.clip(airlight, 95.0, 225.0).astype(np.float32)


def _apply_atmospheric_scattering(image_bgr, distance01, severity, rng=None, strength_range=None):
    """
    Light rain haze using distance-based transmission:
        output = scene * T + airlight * (1 - T)

    The haze target is mostly a local blurred scene color, not a flat gray.
    That keeps rain atmospheric without washing the whole frame into fog.
    """
    img = _as_uint8_bgr(image_bgr).astype(np.float32)
    h, w = img.shape[:2]
    s = _severity_value(severity)

    d = np.asarray(distance01, dtype=np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    d = np.clip(d, 0.0, 1.0)

    if strength_range is None:
        beta = 0.012 + 0.135 * s
        exposure_loss = 0.006 + 0.020 * s
    else:
        low, high = strength_range
        if high < low:
            raise ValueError("strength_range must be ordered as (min_value, max_value).")
        legacy_strength = float((low + high) * 0.5) / 255.0
        beta = 0.010 + 0.80 * legacy_strength
        exposure_loss = np.clip(legacy_strength * 0.22, 0.002, 0.035)

    depth_power = np.power(d, 1.75)
    far_gate = _smoothstep(0.22, 0.92, d)
    haze_amount = (1.0 - np.exp(-beta * depth_power)) * far_gate
    haze_amount = np.clip(haze_amount, 0.0, 0.035 + 0.085 * s)
    transmission = (1.0 - haze_amount)[..., None]
    exposure = 1.0 - exposure_loss * far_gate[..., None]

    img_lin = _srgb_to_linear01(img)
    blur_sigma = max(2.0, min(h, w) * (0.012 + 0.010 * s))
    local_haze_bgr = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    airlight_bgr = _estimate_haze_airlight_bgr(img).reshape(1, 1, 3)
    airlight_mix = (0.10 + 0.16 * s) * far_gate[..., None]
    haze_target_bgr = local_haze_bgr * (1.0 - airlight_mix) + airlight_bgr * airlight_mix
    haze_target_lin = _srgb_to_linear01(haze_target_bgr)

    out_lin = img_lin * transmission * exposure + haze_target_lin * (1.0 - transmission)

    lum = (
        0.0722 * out_lin[..., 0]
        + 0.7152 * out_lin[..., 1]
        + 0.2126 * out_lin[..., 2]
    )[..., None]
    desat = ((0.006 + 0.032 * s) * depth_power * far_gate)[..., None]
    out_lin = out_lin * (1.0 - desat) + lum * desat

    original_lum = (
        0.0722 * img_lin[..., 0]
        + 0.7152 * img_lin[..., 1]
        + 0.2126 * img_lin[..., 2]
    )[..., None]
    out_lum = (
        0.0722 * out_lin[..., 0]
        + 0.7152 * out_lin[..., 1]
        + 0.2126 * out_lin[..., 2]
    )[..., None]
    preserve = 0.65 * haze_amount[..., None]
    target_lum = original_lum * (1.0 - 0.045 * s * far_gate[..., None])
    out_lin *= (1.0 - preserve) + preserve * target_lum / (out_lum + 1e-6)

    if rng is not None and h > 2 and w > 2:
        noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), max(4.0, min(h, w) * 0.045))
        noise = noise / (np.max(np.abs(noise)) + 1e-6)
        veil_variation = 1.0 + (0.004 + 0.010 * s) * noise[..., None] * far_gate[..., None]
        out_lin = np.clip(out_lin * veil_variation, 0.0, 1.0)

    return _linear01_to_srgb255(out_lin).astype(np.uint8)


def _composite_rain_delta_linear(image_bgr, rain_alpha, rain_delta):
    base = _as_uint8_bgr(image_bgr).astype(np.float32)
    alpha = np.clip(np.asarray(rain_alpha, dtype=np.float32), 0.0, 1.0)[..., None]
    delta = np.asarray(rain_delta, dtype=np.float32)

    if delta.ndim == 2:
        tint = np.array([0.96, 1.00, 1.04], dtype=np.float32).reshape(1, 1, 3)
        delta_rgb = delta[..., None] * tint
    else:
        delta_rgb = delta

    target = np.clip(base + delta_rgb, 0.0, 255.0)

    base_lin = _srgb_to_linear01(base)
    target_lin = _srgb_to_linear01(target)
    out_lin = base_lin * (1.0 - alpha) + target_lin * alpha

    return _linear01_to_srgb255(out_lin).astype(np.uint8)


def _render_simple_realistic_rain(image_bgr, rng, severity_value, angle_deg=105):
    """
    Non-depth rain made from soft, tapered motion streaks.

    This intentionally avoids the depth-aware renderer. It only improves the
    simple baseline's 2D streak shape and keeps visibility restrained.
    """
    img = _as_uint8_bgr(image_bgr)
    h, w = img.shape[:2]
    s = float(severity_value)
    area_scale = h * w / 1000.0
    min_dim = float(min(h, w))
    luma = _linear_luminance01(img)

    layer_specs = [
        {
            "count": int(area_scale * (0.45 + 1.55 * s)),
            "length": min_dim * (0.025 + 0.042 * s),
            "width": 1.15 + 0.75 * s,
            "opacity": 0.050 + 0.095 * s,
            "delta": 64.0 + 42.0 * s,
            "shadow": 4.0 + 5.0 * s,
            "blur": 0.55 + 0.35 * s,
            "angle_jitter": 6.0,
        },
        {
            "count": int(area_scale * (1.05 + 4.00 * s)),
            "length": min_dim * (0.012 + 0.027 * s),
            "width": 0.70 + 0.35 * s,
            "opacity": 0.035 + 0.075 * s,
            "delta": 48.0 + 36.0 * s,
            "shadow": 3.0 + 4.0 * s,
            "blur": 0.25 + 0.25 * s,
            "angle_jitter": 4.0,
        },
        {
            "count": int(area_scale * (0.55 + 2.45 * s)),
            "length": min_dim * (0.006 + 0.014 * s),
            "width": 0.45 + 0.20 * s,
            "opacity": 0.018 + 0.040 * s,
            "delta": 36.0 + 26.0 * s,
            "shadow": 1.5 + 3.0 * s,
            "blur": 0.15 + 0.18 * s,
            "angle_jitter": 3.0,
        },
    ]

    total_tau = np.zeros((h, w), dtype=np.float32)
    total_delta_tau = np.zeros((h, w), dtype=np.float32)

    for spec in layer_specs:
        count = max(0, int(spec["count"]))
        if count <= 0:
            continue

        local_angle = float(angle_deg + rng.normal(0.0, spec["angle_jitter"]))
        local_length = max(3.0, float(spec["length"]) * rng.uniform(0.90, 1.12))
        local_width = max(0.42, float(spec["width"]) * rng.uniform(0.90, 1.10))
        kernel = _make_motion_streak_kernel(local_length, local_angle, local_width, taper_power=1.75)

        xs = rng.integers(0, w, size=count, dtype=np.int32)
        ys = rng.integers(0, h, size=count, dtype=np.int32)

        seed_tau = np.zeros((h, w), dtype=np.float32)
        seed_delta_tau = np.zeros((h, w), dtype=np.float32)

        seed_luma = np.clip(luma[ys, xs], 0.0, 1.0)
        contrast = 0.26 + 0.72 * np.power(1.0 - seed_luma, 0.78)
        deltas = float(spec["delta"]) * contrast - float(spec["shadow"]) * np.power(seed_luma, 1.20)
        deltas += rng.normal(0.0, 2.2 + 2.8 * s, size=count)
        deltas = np.clip(deltas, -8.0, 78.0).astype(np.float32)

        opacities = float(spec["opacity"]) * rng.lognormal(mean=0.0, sigma=0.20, size=count)
        opacities = np.clip(opacities, 0.0015, 0.180).astype(np.float32)

        np.add.at(seed_tau, (ys, xs), opacities)
        np.add.at(seed_delta_tau, (ys, xs), opacities * deltas)

        layer_tau = cv2.filter2D(seed_tau, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
        layer_delta_tau = cv2.filter2D(seed_delta_tau, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)

        blur = float(spec["blur"])
        if blur > 0.05:
            layer_tau = cv2.GaussianBlur(layer_tau, (0, 0), blur)
            layer_delta_tau = cv2.GaussianBlur(layer_delta_tau, (0, 0), blur)

        total_tau += layer_tau
        total_delta_tau += layer_delta_tau

    alpha = 1.0 - np.exp(-np.clip(total_tau * (2.10 + 1.20 * s), 0.0, 3.0))
    delta = total_delta_tau / (total_tau + 1e-6)
    delta = np.where(total_tau > 1e-7, delta, 0.0)

    return np.clip(alpha, 0.0, 0.42).astype(np.float32), np.clip(delta, -8.0, 112.0).astype(np.float32)


def add_simple_rain(image_bgr, severity="medium", seed=0, angle_deg=105):
    """
    Baseline renderer: add 2D rain without using depth.

    Uses soft, tapered motion streaks with restrained opacity so the simple
    rain remains visible without looking like painted white scratches.
    """
    img = _as_uint8_bgr(image_bgr)
    s = _severity_value(severity)
    rain_s = _rain_visibility_severity(s)
    rng = np.random.default_rng(seed)

    rain_alpha, rain_delta = _render_simple_realistic_rain(
        img,
        rng=rng,
        severity_value=rain_s,
        angle_deg=angle_deg,
    )

    darken = _rain_fog_darkening_map(
        img.shape,
        rng,
        severity,
        strength_range=(0.0, 0.70 + 0.80 * s),
    )
    weathered = np.maximum(img.astype(np.float32) - darken[..., None], 0.0).astype(np.uint8)

    return _composite_rain_delta_linear(weathered, rain_alpha, rain_delta)


def add_depth_aware_rain(
    image_bgr,
    depth_map,
    severity="medium",
    seed=0,
    angle_deg=105,
    close_is_high=True,
    add_veil=True,
    fog_darkening_range=None,
):
    """
    Layered depth-aware rain renderer.

    Design:
        near/mid/far/background rain layers are rendered at fixed apparent
        depths, then clipped against the scene depth so far rain does not paint
        over foreground objects.
        strokes use tapered anisotropic motion kernels instead of hard lines.
        contrast is background-aware, with stronger highlights on dark regions
        and lower contrast over bright sky.
        add_veil enables distance-based atmospheric scattering rather than
        subtractive darkening.
    """
    img = _as_uint8_bgr(image_bgr)
    s = _severity_value(severity)
    rain_s = _rain_visibility_severity(s)
    rng = np.random.default_rng(seed)

    distance01 = normalize_depth_for_rain(
        depth_map,
        image_shape=img.shape,
        close_is_high=close_is_high,
    )

    if add_veil:
        weathered = _apply_atmospheric_scattering(
            img,
            distance01,
            severity=s,
            rng=rng,
            strength_range=fog_darkening_range,
        )
    else:
        weathered = img.copy()

    rain_alpha, rain_delta = _render_layered_depth_rain(
        weathered,
        distance01,
        rng=rng,
        severity_value=rain_s,
        angle_deg=angle_deg,
    )

    return _composite_rain_delta_linear(weathered, rain_alpha, rain_delta)


def show_rain_comparison(image_bgr, depth_map=None, severity="medium", seed=7, close_is_high=True):
    if plt is None:
        raise RuntimeError("matplotlib is required for show_rain_comparison().")

    img = _as_uint8_bgr(image_bgr)

    if depth_map is None:
        depth_map = estimate_depth(img)

    simple = add_simple_rain(img, severity=severity, seed=seed)

    depth_rain = add_depth_aware_rain(
        img,
        depth_map,
        severity=severity,
        seed=seed,
        close_is_high=close_is_high,
    )

    distance01 = normalize_depth_for_rain(
        depth_map,
        image_shape=img.shape,
        close_is_high=close_is_high,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Clean image")
    axes[0].axis("off")

    axes[1].imshow(distance01, cmap="magma")
    axes[1].set_title("Estimated distance map")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(simple, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Simple 2D rain")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(depth_rain, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Layered depth-aware rain")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "clean": img,
        "depth": depth_map,
        "distance01": distance01,
        "simple_rain": simple,
        "depth_aware_rain": depth_rain,
    }


def run_rain_renderer_self_tests(image_bgr, depth_map=None):
    """
    Lightweight checks without assert statements.
    """
    img = _as_uint8_bgr(image_bgr)

    if depth_map is None:
        h, w = img.shape[:2]
        depth_map = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))

    simple_a = add_simple_rain(img, severity="medium", seed=123)
    simple_b = add_simple_rain(img, severity="medium", seed=123)

    depth_a = add_depth_aware_rain(img, depth_map, severity="medium", seed=123)
    depth_b = add_depth_aware_rain(img, depth_map, severity="medium", seed=123)

    checks = {
        "simple_shape_ok": simple_a.shape == img.shape,
        "depth_shape_ok": depth_a.shape == img.shape,
        "simple_dtype_ok": simple_a.dtype == np.uint8,
        "depth_dtype_ok": depth_a.dtype == np.uint8,
        "simple_seed_deterministic": np.array_equal(simple_a, simple_b),
        "depth_seed_deterministic": np.array_equal(depth_a, depth_b),
        "depth_finite": np.isfinite(depth_map).all(),
        "simple_range_ok": simple_a.min() >= 0 and simple_a.max() <= 255,
        "depth_range_ok": depth_a.min() >= 0 and depth_a.max() <= 255,
    }

    distance01 = normalize_depth_for_rain(depth_map, image_shape=img.shape)
    close_half = depth_a[:, : depth_a.shape[1] // 2].astype(np.float32)
    far_half = depth_a[:, depth_a.shape[1] // 2 :].astype(np.float32)
    if distance01[:, : distance01.shape[1] // 2].mean() < distance01[:, distance01.shape[1] // 2 :].mean():
        checks["far_side_weathered_more"] = far_half.std() >= close_half.std() * 0.55

    failed = [name for name, ok in checks.items() if not ok]

    if failed:
        raise RuntimeError(f"Rain renderer checks failed: {failed}")

    print("Rain renderer checks passed.")
