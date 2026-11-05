"""
INKBOUND — Cinematic Effects and Compositing
File: src/effects.py

Dependencies:
    Pillow
    NumPy
    src/config.py
    src/characters.py
    src/choreography.py
    src/typography.py

Effects:
    - Camera zoom, pan and shake
    - Monochrome film grain
    - Cinematic vignette
    - Ink intrusion and transitions
    - Directional speed lines
    - Impact flashes
    - Text-safe masking
    - Loop-boundary fade
    - Deterministic compositing

All effects are evaluated from absolute time.
No frame-to-frame state is required.

IMPORTANT:
    Camera transforms must be applied exactly once.

    Use apply_camera_to_layer() for world layers
    that have not already been camera-transformed.

    Characters rendered using
    choreography.apply_camera_to_character()
    are already in screen space.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    from .config import CONFIG, OUTPUT_DIR, PALETTE, SEED
    from .characters import Vec2, clamp, smoothstep
    from .choreography import Camera, TimelineState, evaluate_timeline, loop_time
    from .typography import text_safe_zones
except ImportError:  # pragma: no cover - direct script execution
    from config import CONFIG, OUTPUT_DIR, PALETTE, SEED  # type: ignore
    from characters import Vec2, clamp, smoothstep  # type: ignore
    from choreography import Camera, TimelineState, evaluate_timeline, loop_time  # type: ignore
    from typography import text_safe_zones  # type: ignore

__all__ = [
    "background_frame",
    "apply_camera_to_layer",
    "vignette_mask",
    "apply_vignette",
    "grain_layer",
    "apply_grain",
    "ink_layer",
    "speed_line_layer",
    "flash_layer",
    "apply_safe_zone_mask",
    "loop_fade_layer",
    "compose_scene",
    "PREVIEW_TIMES",
]

PREVIEW_TIMES: Tuple[float, ...] = (1.5, 3.4, 5.48, 6.5, 10.2, 11.9)

Mask = Tuple[int, int, int, int]


# --------------------------------------------------------------------------
# Background
# --------------------------------------------------------------------------


def background_frame(size: Tuple[int, int] = None) -> Image.Image:
    """Flat #080808 canvas with the faintest top-lit gradient."""
    size = size or CONFIG.size()
    width, height = size
    top = np.array(PALETTE.background, dtype=np.float32)
    bottom = np.array(tuple(min(255, c + 6) for c in PALETTE.background), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    gradient = top[None, None, :] * (1.0 - ramp) + bottom[None, None, :] * ramp
    gradient = np.repeat(gradient, width, axis=1)
    return Image.fromarray(gradient.astype(np.uint8), "RGB").convert("RGBA")


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


def _affine_inverse(camera: Camera, size: Tuple[int, int]) -> Tuple[float, float, float, float, float, float]:
    """
    Coefficients mapping screen pixels back to source pixels.

    Pillow's AFFINE transform reads: src = A * dst + offset, which is
    exactly the inverse of choreography's world_to_screen().
    """
    width, height = size
    cx, cy = width * 0.5, height * 0.5
    zoom = camera.zoom if abs(camera.zoom) > 1e-9 else 1.0
    theta = camera.rotation
    cos_t = math.cos(theta) / zoom
    sin_t = math.sin(theta) / zoom

    # M = (1/zoom) * R(-theta)
    m00, m01 = cos_t, sin_t
    m10, m11 = -sin_t, cos_t

    sx, sy = camera.shake.x, camera.shake.y
    px, py = camera.pan.x, camera.pan.y

    a = m00
    b = m01
    c = -m00 * (cx + sx) - m01 * (cy + sy) + cx + px
    d = m10
    e = m11
    f = -m10 * (cx + sx) - m11 * (cy + sy) + cy + py
    return (a, b, c, d, e, f)


def apply_camera_to_layer(
    layer: Image.Image,
    camera: Camera,
    size: Tuple[int, int] = None,
) -> Image.Image:
    """
    Transform a world-space RGBA layer into screen space.

    Call this exactly once for any layer that has not already gone
    through choreography.apply_camera_to_character().  Identity
    cameras return the input unchanged, byte for byte.
    """
    size = size or layer.size
    if camera.is_identity:
        if layer.size == size:
            return layer
        return layer.resize(size, Image.BICUBIC)

    coeffs = _affine_inverse(camera, size)
    transformed = layer.transform(
        size,
        Image.AFFINE,
        coeffs,
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    return transformed


# --------------------------------------------------------------------------
# Vignette
# --------------------------------------------------------------------------

_VIGNETTE_CACHE: dict = {}


def vignette_mask(size: Tuple[int, int] = None, strength: float = 0.62) -> Image.Image:
    """Luminance mask: white in the centre, dark at the corners."""
    size = size or CONFIG.size()
    key = (size, round(strength, 3))
    if key in _VIGNETTE_CACHE:
        return _VIGNETTE_CACHE[key]

    width, height = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xs - width * 0.5) / max(1.0, width * 0.5)
    ny = (ys - height * 0.5) / max(1.0, height * 0.5)
    radius = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
    falloff = np.clip((radius - 0.35) / 0.65, 0.0, 1.0)
    value = 1.0 - strength * (falloff ** 1.6)
    mask = (np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(mask, "L").convert("RGBA")
    _VIGNETTE_CACHE[key] = image
    return image


def apply_vignette(frame: Image.Image, strength: float = 0.62) -> Image.Image:
    if strength <= 0.0:
        return frame
    mask = vignette_mask(frame.size, strength)
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    # Pillow's composite() reads the alpha band when handed an RGBA mask,
    # and our mask carries its luminance in RGB — so blend with it as L.
    return Image.composite(
        frame,
        Image.new("RGBA", frame.size, (0, 0, 0, 255)),
        mask.convert("L"),
    )


# --------------------------------------------------------------------------
# Film grain
# --------------------------------------------------------------------------


def grain_layer(
    size: Tuple[int, int] = None,
    time: float = 0.0,
    strength: float = 0.06,
    frame_index: Optional[int] = None,
) -> Image.Image:
    """
    Monochrome grain, re-seeded from absolute time.

    Deterministic: the same time always yields the same noise field,
    so out-of-order rendering still matches.
    """
    size = size or CONFIG.size()
    if strength <= 0.0:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    index = CONFIG.frame_index(time) if frame_index is None else int(frame_index)
    rng = np.random.default_rng(SEED + index)
    width, height = size
    noise = rng.normal(128.0, 42.0, size=(height, width)).clip(0, 255).astype(np.uint8)
    gray = Image.fromarray(noise, "L")
    alpha = gray.point(lambda v: int(abs(v - 128) * strength * 4.0))
    rgb = Image.merge("RGB", (gray, gray, gray))
    return Image.merge("RGBA", (*rgb.split(), alpha))


def apply_grain(frame: Image.Image, time: float, strength: float = 0.06) -> Image.Image:
    if strength <= 0.0:
        return frame
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    layer = grain_layer(frame.size, time, strength)
    frame = frame.copy()
    frame.alpha_composite(layer)
    return frame


# --------------------------------------------------------------------------
# Ink intrusion
# --------------------------------------------------------------------------


def ink_layer(
    size: Tuple[int, int] = None,
    time: float = 0.0,
    amount: float = 0.0,
    seed: int = SEED,
) -> Image.Image:
    """
    Dark ink tendrils crawling in from the frame edges.

    `amount` comes from the choreography ink curve and peaks at every
    scene cut, so the ink reads as a transition wipe.
    """
    size = size or CONFIG.size()
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if amount <= 0.01:
        return layer

    draw = ImageDraw.Draw(layer)
    width, height = size
    rng_seed = seed + int(time * 1000) % 100000

    import random

    rng = random.Random(rng_seed)
    tendrils = 26
    for index in range(tendrils):
        edge = index % 4
        along = rng.random()
        reach = amount * rng.uniform(0.35, 1.0)
        if edge == 0:      # top
            x, y = along * width, -10
            dx, dy = rng.uniform(-0.4, 0.4), 1.0
        elif edge == 1:    # right
            x, y = width + 10, along * height
            dx, dy = -1.0, rng.uniform(-0.4, 0.4)
        elif edge == 2:    # bottom
            x, y = along * width, height + 10
            dx, dy = rng.uniform(-0.4, 0.4), -1.0
        else:              # left
            x, y = -10, along * height
            dx, dy = 1.0, rng.uniform(-0.4, 0.4)

        length = reach * max(width, height) * rng.uniform(0.35, 0.8)
        steps = 8
        points = []
        px, py = x, y
        for step in range(steps):
            px += dx * length / steps + rng.uniform(-14, 14) * amount
            py += dy * length / steps + rng.uniform(-14, 14) * amount
            points.append((px, py))
        thickness = int(6 + 34 * amount)
        draw.line(points, fill=(0, 0, 0, int(clamp(230 * amount, 0, 255))),
                  width=thickness, joint="curve")
        radius = thickness
        draw.ellipse([points[-1][0] - radius, points[-1][1] - radius,
                      points[-1][0] + radius, points[-1][1] + radius],
                     fill=(0, 0, 0, int(clamp(210 * amount, 0, 255))))

    return layer.filter(ImageFilter.GaussianBlur(radius=max(1.0, 8.0 * (1.0 - amount))))


# --------------------------------------------------------------------------
# Speed lines
# --------------------------------------------------------------------------


def speed_line_layer(
    size: Tuple[int, int] = None,
    amount: float = 0.0,
    time: float = 0.0,
    origin: Tuple[float, float] = None,
) -> Image.Image:
    """Directional streaks radiating from the clash point."""
    size = size or CONFIG.size()
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if amount <= 0.02:
        return layer

    draw = ImageDraw.Draw(layer)
    width, height = size
    ox, oy = origin or (width * 0.5, height * 0.38)
    import random

    rng = random.Random(SEED + int(time * 60))
    lines = int(34 * amount) + 6
    for _ in range(lines):
        angle = rng.uniform(0.0, math.tau)
        inner = rng.uniform(90, 260)
        outer = inner + rng.uniform(80, 320) * amount
        x0 = ox + math.cos(angle) * inner
        y0 = oy + math.sin(angle) * inner
        x1 = ox + math.cos(angle) * outer
        y1 = oy + math.sin(angle) * outer
        alpha = int(clamp(150 * amount * rng.uniform(0.4, 1.0), 0, 255))
        draw.line([(x0, y0), (x1, y1)],
                  fill=(PALETTE.ink[0], PALETTE.ink[1], PALETTE.ink[2], alpha),
                  width=rng.choice((1, 1, 2)))
    return layer


# --------------------------------------------------------------------------
# Flash
# --------------------------------------------------------------------------


def flash_layer(size: Tuple[int, int] = None, amount: float = 0.0) -> Image.Image:
    """Full-frame impact flash. Screen space, never camera-transformed."""
    size = size or CONFIG.size()
    if amount <= 0.004:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    alpha = int(clamp(amount * 200, 0, 255))
    return Image.new("RGBA", size, PALETTE.ink + (alpha,))


# --------------------------------------------------------------------------
# Text-safe masking
# --------------------------------------------------------------------------


def apply_safe_zone_mask(
    layer: Image.Image,
    zones: Iterable[Mask] = None,
    feather: int = 10,
    strength: float = 1.0,
) -> Image.Image:
    """
    Erase (or fade) layer content inside the text-safe rectangles.

    This is how the tree and fighters are kept out of the profile
    copy: the typography engine only *reports* the zones, the
    composition stage does the clipping.
    """
    if zones is None:
        return layer
    rects = [r for r in zones if r[2] > r[0] and r[3] > r[1]]
    if not rects:
        return layer
    if layer.mode != "RGBA":
        layer = layer.convert("RGBA")

    alpha = layer.getchannel("A")
    cutter = Image.new("L", layer.size, 0)
    draw = ImageDraw.Draw(cutter)
    for rect in rects:
        draw.rectangle([rect[0], rect[1], rect[2] - 1, rect[3] - 1], fill=255)
    if feather > 0:
        cutter = cutter.filter(ImageFilter.GaussianBlur(radius=feather))

    # scale controls how hard the mask bites: 1.0 erases completely.
    factor = clamp(strength, 0.0, 1.0)
    scaled = cutter.point(lambda v: int(v * factor))
    new_alpha = Image.fromarray(
        np.clip(
            np.asarray(alpha, dtype=np.int16) - np.asarray(scaled, dtype=np.int16),
            0,
            255,
        ).astype(np.uint8),
        "L",
    )
    result = layer.copy()
    result.putalpha(new_alpha)
    return result


# --------------------------------------------------------------------------
# Loop fade
# --------------------------------------------------------------------------


def loop_fade_layer(size: Tuple[int, int] = None, amount: float = 0.0) -> Image.Image:
    """Black wash across the 12s -> 0s seam."""
    size = size or CONFIG.size()
    if amount <= 0.004:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    return Image.new("RGBA", size, (0, 0, 0, int(clamp(amount * 255, 0, 255))))


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def _onto(base: Image.Image, layer: Optional[Image.Image]) -> None:
    if layer is None:
        return
    if layer.size != base.size:
        layer = layer.resize(base.size, Image.LANCZOS)
    if layer.mode != "RGBA":
        layer = layer.convert("RGBA")
    base.alpha_composite(layer)


def compose_scene(
    background: Image.Image,
    world_layer: Optional[Image.Image] = None,
    character_layer: Optional[Image.Image] = None,
    particle_layer: Optional[Image.Image] = None,
    typography_layer: Optional[Image.Image] = None,
    time: float = 0.0,
    world_is_camera_transformed: bool = False,
    safe_zones: Sequence[Mask] = None,
    state: Optional[TimelineState] = None,
    grain_strength: float = 0.055,
    vignette_strength: float = 0.62,
    draft: bool = False,
) -> Image.Image:
    """
    Build one finished RGBA frame.

    Pipeline
    --------
    1. background
    2. world layer  -> camera applied once, then text-safe masked
    3. ink intrusion + speed lines (screen space)
    4. character layer (already camera-transformed) + particles
    5. typography (screen space, never shaken)
    6. post: vignette, grain, flash, loop fade
    """
    time = loop_time(time)
    state = state or evaluate_timeline(time)
    size = background.size if background is not None else CONFIG.size()

    frame = background.convert("RGBA") if background is not None else background_frame(size)

    # 2. world layer
    if world_layer is not None:
        world = world_layer
        if not world_is_camera_transformed:
            world = apply_camera_to_layer(world, state.camera, size)
        if safe_zones is None:
            safe_zones = text_safe_zones(time)
        world = apply_safe_zone_mask(world, safe_zones, feather=12, strength=0.92)
        _onto(frame, world)

    # 3. screen-space atmosphere
    _onto(frame, ink_layer(size, time, state.ink))
    _onto(frame, speed_line_layer(size, state.speed_lines, time))

    # 4. actors and particles
    _onto(frame, character_layer)
    _onto(frame, particle_layer)

    # 5. typography, fixed to the canvas
    _onto(frame, typography_layer)

    # 6. post-processing
    frame = apply_vignette(frame, vignette_strength)
    if not draft:
        frame = apply_grain(frame, time, grain_strength)
    _onto(frame, flash_layer(size, state.flash))
    _onto(frame, loop_fade_layer(size, state.loop_fade))

    return frame


# --------------------------------------------------------------------------
# Standalone preview
# --------------------------------------------------------------------------


def render_preview(
    times: Sequence[float] = PREVIEW_TIMES,
    panel_size: Tuple[int, int] = (480, 270),
    out_path: Optional[Path] = None,
) -> Image.Image:
    """Six timestamps: camera, ink, grain, impact and the loop fade."""
    import random

    columns, rows = 3, 2
    sheet = Image.new(
        "RGB",
        (panel_size[0] * columns, panel_size[1] * rows),
        CONFIG.background,
    )
    size = CONFIG.size()

    for index, moment in enumerate(times):
        state = evaluate_timeline(moment)

        # A stand-in world layer so the camera pass is visible.
        world = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(world)
        rng = random.Random(SEED + index)
        for _ in range(60):
            x = rng.uniform(0, size[0])
            y = rng.uniform(0, size[1])
            draw.line([(x, y), (x + rng.uniform(-40, 40), y + rng.uniform(-40, 40))],
                      fill=(90, 90, 90, 160), width=1)
        draw.rectangle([0, int(size[1] * 0.72), size[0], size[1]],
                       fill=(30, 30, 30, 255))

        frame = compose_scene(
            background=background_frame(size),
            world_layer=world,
            time=moment,
            state=state,
            grain_strength=0.07,
        )

        draw2 = ImageDraw.Draw(frame)
        draw2.text((8, 6), f"t={moment:.2f}s {state.scene.upper()}", fill=PALETTE.ink_dim)
        draw2.text((8, 22),
                   f"zoom={state.camera.zoom:.2f} ink={state.ink:.2f} "
                   f"flash={state.flash:.2f} fade={state.loop_fade:.2f}",
                   fill=PALETTE.ink_faint)

        panel = frame.convert("RGB").resize(panel_size, Image.LANCZOS)
        sheet.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))

    path = out_path or (OUTPUT_DIR / "effects_preview.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return sheet


# --------------------------------------------------------------------------
# Standalone tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    size = CONFIG.size()

    # Background.
    bg = background_frame(size)
    assert bg.size == size and bg.mode == "RGBA"
    assert background_frame(size).tobytes() == bg.tobytes()

    # Camera identity is a no-op.
    identity_layer = Image.new("RGBA", size, (10, 20, 30, 255))
    assert apply_camera_to_layer(identity_layer, Camera()).tobytes() == identity_layer.tobytes()

    # Zoom keeps the canvas size and moves content off the edges.
    patterned = Image.new("RGBA", size, (0, 0, 0, 0))
    patterned.putpixel((10, 10), (255, 0, 0, 255))
    patterned.putpixel((size[0] - 10, size[1] - 10), (0, 255, 0, 255))
    zoomed = apply_camera_to_layer(patterned, Camera(zoom=2.0), size)
    assert zoomed.size == size
    assert zoomed.tobytes() != patterned.tobytes()
    assert zoomed.getpixel((size[0] // 2, size[1] // 2))[3] == 0, (
        "a 2x zoom pushes the corner markers out of frame"
    )

    # Inverse mapping agrees with choreography's forward mapping.
    cam = Camera(zoom=1.4, pan=Vec2(12.0, -8.0), shake=Vec2(3.0, 4.0))
    point = Vec2(200.0, 300.0)
    screen = cam.world_to_screen(point, size)
    coeffs = _affine_inverse(cam, size)
    back_x = coeffs[0] * screen.x + coeffs[1] * screen.y + coeffs[2]
    back_y = coeffs[3] * screen.x + coeffs[4] * screen.y + coeffs[5]
    assert abs(back_x - point.x) < 1e-6 and abs(back_y - point.y) < 1e-6

    # Vignette darkens corners, not the centre.
    vig = vignette_mask(size, 0.62)
    centre_value = vig.getpixel((size[0] // 2, size[1] // 2))[0]
    corner_value = vig.getpixel((2, 2))[0]
    assert centre_value > corner_value
    assert centre_value == 255
    assert vig.tobytes() == vignette_mask(size, 0.62).tobytes()

    dark = apply_vignette(Image.new("RGBA", size, (255, 255, 255, 255)), 0.8)
    assert dark.getpixel((2, 2))[0] < 255
    assert dark.getpixel((size[0] // 2, size[1] // 2))[0] == 255

    # Grain: deterministic per time, different across times.
    g1 = grain_layer(size, 1.0)
    g2 = grain_layer(size, 1.0)
    g3 = grain_layer(size, 2.0)
    assert g1.tobytes() == g2.tobytes()
    assert g1.tobytes() != g3.tobytes()
    assert grain_layer(size, 1.0, strength=0.0).getbbox() is None
    no_grain = apply_grain(Image.new("RGBA", size, (10, 10, 10, 255)), 1.0, 0.0)
    assert no_grain.getpixel((0, 0))[:3] == (10, 10, 10)

    # Ink intrusion tracks the choreography curve.
    assert ink_layer(size, 2.0, 0.0).getbbox() is None
    light_ink = ink_layer(size, 2.9, 0.3)
    heavy_ink = ink_layer(size, 2.99, 0.95)
    assert light_ink.getbbox() is not None and heavy_ink.getbbox() is not None
    assert evaluate_timeline(2.99).ink > evaluate_timeline(2.5).ink

    # Speed lines.
    assert speed_line_layer(size, 0.0, 1.0).getbbox() is None
    assert speed_line_layer(size, 0.8, 1.0).getbbox() is not None

    # Flash.
    assert flash_layer(size, 0.0).getbbox() is None
    assert flash_layer(size, 0.7).getpixel((0, 0))[3] > 100

    # Safe-zone masking erases inside, preserves outside.
    zone = (100, 100, 300, 240)
    solid = Image.new("RGBA", size, (255, 255, 255, 255))
    masked = apply_safe_zone_mask(solid, [zone], feather=0)
    assert masked.getpixel((200, 170))[3] == 0, "zone interior must be erased"
    assert masked.getpixel((600, 400))[3] == 255, "outside the zone must survive"
    assert apply_safe_zone_mask(solid, []).tobytes() == solid.tobytes()
    partial = apply_safe_zone_mask(solid, [zone], feather=0, strength=0.5)
    assert 0 < partial.getpixel((200, 170))[3] < 255

    # Loop fade.
    assert loop_fade_layer(size, 0.0).getbbox() is None
    assert loop_fade_layer(size, 1.0).getpixel((0, 0))[3] == 255
    assert evaluate_timeline(0.0).loop_fade > 0.9
    assert evaluate_timeline(11.99).loop_fade > 0.9
    assert evaluate_timeline(6.0).loop_fade == 0.0

    # Compose: correct size, deterministic, out-of-order safe.
    a = compose_scene(background_frame(size), time=6.5)
    b = compose_scene(background_frame(size), time=6.5)
    assert a.size == size and a.mode == "RGBA"
    assert a.tobytes() == b.tobytes()

    world = Image.new("RGBA", size, (40, 40, 40, 255))
    forward = [compose_scene(background_frame(size), world_layer=world, time=t).tobytes()
               for t in (1.0, 5.48, 11.9)]
    backward = [compose_scene(background_frame(size), world_layer=world, time=t).tobytes()
                for t in reversed((1.0, 5.48, 11.9))][::-1]
    assert forward == backward

    # world_is_camera_transformed must change the result (no double pass).
    cam_state = evaluate_timeline(6.5)
    raw = compose_scene(background_frame(size), world_layer=world, time=6.5,
                        world_is_camera_transformed=False, state=cam_state)
    pre = compose_scene(background_frame(size),
                        world_layer=apply_camera_to_layer(world, cam_state.camera, size),
                        time=6.5, world_is_camera_transformed=True, state=cam_state)
    assert raw.tobytes() == pre.tobytes()

    # Draft mode skips grain.
    draft = compose_scene(background_frame(size), time=6.5, draft=True)
    full = compose_scene(background_frame(size), time=6.5, draft=False)
    assert draft.tobytes() != full.tobytes()

    # Preview.
    sheet = render_preview()
    assert sheet.size == (480 * 3, 270 * 2)
    assert (OUTPUT_DIR / "effects_preview.png").exists()


if __name__ == "__main__":
    _self_test()
    print("All cinematic effects tests passed.")
    print("INKBOUND effects engine loaded.")
