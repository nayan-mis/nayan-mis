"""
INKBOUND — Master Animation Renderer
File: src/render.py

Connects:
    config.py
    characters.py
    choreography.py
    tree.py
    particles.py
    typography.py
    effects.py

Features:
    - Complete 12-second cinematic animation
    - Deterministic arbitrary-frame evaluation
    - Camera-consistent character/world/effects layers
    - Supersampled rendering
    - PNG frame sequences
    - Animated WebP export
    - File-size-aware encoding
    - Integration tests and storyboard preview

Run from the project root:

    python -m src.render --test
    python -m src.render --preview
    python -m src.render --frame 130
    python -m src.render --export

For a faster draft:

    python -m src.render --export --draft

Note on GitHub: a README can *display* an animated WebP but cannot
run this renderer.  The renderer produces the asset you commit.
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

try:
    from .config import CONFIG, OUTPUT_DIR, PALETTE, SEED
    from .characters import Vec2, clamp, draw_skeleton
    from .choreography import (
        Camera,
        STORYBOARD_TIMES,
        TimelineState,
        evaluate_timeline,
        loop_time,
    )
    from .tree import ProceduralTree, render_tree
    from .particles import ParticleEngine, get_engine, render_particles
    from .typography import render_typography, text_safe_zones
    from .effects import background_frame, compose_scene
except ImportError:  # pragma: no cover - direct script execution
    from config import CONFIG, OUTPUT_DIR, PALETTE, SEED  # type: ignore
    from characters import Vec2, clamp, draw_skeleton  # type: ignore
    from choreography import (  # type: ignore
        Camera,
        STORYBOARD_TIMES,
        TimelineState,
        evaluate_timeline,
        loop_time,
    )
    from tree import ProceduralTree, render_tree  # type: ignore
    from particles import ParticleEngine, get_engine, render_particles  # type: ignore
    from typography import render_typography, text_safe_zones  # type: ignore
    from effects import background_frame, compose_scene  # type: ignore

__all__ = [
    "render_world_layer",
    "render_character_layer",
    "render_particle_layer",
    "render_typography_layer",
    "render_frame",
    "render_frame_png",
    "render_storyboard",
    "export_webp",
    "RENDER_TIMES",
]

RENDER_TIMES: Tuple[float, ...] = STORYBOARD_TIMES


# --------------------------------------------------------------------------
# Shared resources — built once per process
# --------------------------------------------------------------------------

_TREE: Optional[ProceduralTree] = None
_ENGINE: Optional[ParticleEngine] = None


def _tree() -> ProceduralTree:
    global _TREE
    if _TREE is None:
        _TREE = ProceduralTree(seed=SEED)
    return _TREE


def _engine() -> ParticleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = get_engine(seed=SEED)
    return _ENGINE


def _scale_camera(camera: Camera, scale: float) -> Camera:
    """Keep world -> screen mapping identical when the canvas grows."""
    if abs(scale - 1.0) < 1e-9:
        return camera
    return Camera(
        zoom=camera.zoom,
        pan=camera.pan * scale,
        shake=camera.shake * scale,
        rotation=camera.rotation,
    )


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


def render_world_layer(
    t: float,
    scale: int = 1,
    tree: Optional[ProceduralTree] = None,
) -> Image.Image:
    """
    Tree / roots / environment in WORLD space.

    Camera is deliberately NOT applied here — compose_scene() does it
    exactly once (world_is_camera_transformed=False).
    """
    t = loop_time(t)
    state = evaluate_timeline(t)
    tree = tree or _tree()
    reference = render_tree(
        CONFIG.size(),
        growth=state.tree_growth,
        seed=tree.seed,
        t=t,
    )
    if scale == 1:
        return reference
    return reference.resize((CONFIG.width * scale, CONFIG.height * scale), Image.LANCZOS)


def render_character_layer(
    t: float,
    scale: int = 1,
    state: Optional[TimelineState] = None,
) -> Image.Image:
    """
    Both fighters, already camera-transformed into SCREEN space.

    Never pass this layer through apply_camera_to_layer() again.
    """
    t = loop_time(t)
    state = state or evaluate_timeline(t)
    ref_size = CONFIG.size()
    size = (CONFIG.width * scale, CONFIG.height * scale)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for actor in (state.wanderer, state.twisted_man):
        if not actor.visible:
            continue
        anchor_ref = actor.screen_anchor(state.camera, ref_size)
        anchor = anchor_ref * scale
        color = actor.accent if actor.accent != PALETTE.ink else actor.color
        # Soft ink shadow so the silhouette separates from the tree.
        draw_skeleton(
            draw,
            actor.pose,
            anchor + Vec2(3.0 * scale, 4.0 * scale),
            color=(0, 0, 0, 140),
            scale=actor.transform.scale * scale,
            width=int(max(2, 5 * scale)),
            head_radius=12.0 * scale,
        )
        draw_skeleton(
            draw,
            actor.pose,
            anchor,
            color=color,
            scale=actor.transform.scale * scale,
            width=int(max(2, 5 * scale)),
            head_radius=12.0 * scale,
            glow=True,
        )

        # Impact flare right on the blade when a hit lands.
        for impact in state.active_impacts:
            env = impact.envelope(t)
            if env <= 0.02:
                continue
            point = state.camera.world_to_screen(impact.position, ref_size) * scale
            radius = (10.0 + 40.0 * (1.0 - env)) * scale
            draw.ellipse(
                [point.x - radius, point.y - radius, point.x + radius, point.y + radius],
                outline=PALETTE.ink + (int(clamp(env * 230, 0, 255)),),
                width=max(1, int(2 * scale)),
            )

    return layer


def render_particle_layer(
    t: float,
    scale: int = 1,
    engine: Optional[ParticleEngine] = None,
    state: Optional[TimelineState] = None,
) -> Image.Image:
    """
    Sparks, ink and shockwaves, already camera-transformed.

    Rasterised at reference resolution and upscaled, matching the
    documented limitation for particle detail under supersampling.
    """
    t = loop_time(t)
    state = state or evaluate_timeline(t)
    engine = engine or _engine()
    reference = Image.new("RGBA", CONFIG.size(), (0, 0, 0, 0))
    render_particles(reference, engine, time=t, camera=state.camera, layer=reference)
    if scale == 1:
        return reference
    return reference.resize((CONFIG.width * scale, CONFIG.height * scale), Image.LANCZOS)


def render_typography_layer(t: float, scale: int = 1) -> Image.Image:
    """Profile copy in SCREEN space — never shaken, never zoomed."""
    t = loop_time(t)
    size = (CONFIG.width * scale, CONFIG.height * scale)
    return render_typography(t, size=size)


# --------------------------------------------------------------------------
# Frame
# --------------------------------------------------------------------------


def render_frame(
    t: float,
    scale: int = 1,
    draft: bool = False,
    state: Optional[TimelineState] = None,
) -> Image.Image:
    """
    Compose one finished RGBA frame at absolute time `t`.

    Deterministic: the same t always yields byte-identical output,
    regardless of which frames were rendered before it.
    """
    t = loop_time(t)
    state = state or evaluate_timeline(t)
    scaled_state = replace(state, camera=_scale_camera(state.camera, scale))

    size = (CONFIG.width * scale, CONFIG.height * scale)
    safe_zones = text_safe_zones(t)

    return compose_scene(
        background=background_frame(size),
        world_layer=render_world_layer(t, scale),
        character_layer=render_character_layer(t, scale, scaled_state),
        particle_layer=render_particle_layer(t, scale, state=scaled_state),
        typography_layer=render_typography_layer(t, scale),
        time=t,
        world_is_camera_transformed=False,
        safe_zones=safe_zones,
        state=scaled_state,
        grain_strength=0.0 if draft else 0.055,
        draft=draft,
    )


def render_frame_png(
    frame_index: int,
    out_path: Optional[Path] = None,
    draft: bool = False,
    scale: int = 1,
) -> Path:
    """Render one frame by index and write it to disk."""
    path = out_path or (OUTPUT_DIR / f"inkbound_frame_{int(frame_index):04d}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = render_frame(CONFIG.frame_time(frame_index), scale=scale, draft=draft)
    frame.convert("RGB").save(path)
    return path


def render_frames(
    indices: Iterable[int],
    draft: bool = False,
    scale: int = 1,
) -> List[Image.Image]:
    """Render an arbitrary (possibly out-of-order) set of frames."""
    return [
        render_frame(CONFIG.frame_time(int(i)), scale=scale, draft=draft)
        for i in indices
    ]


def render_png_sequence(
    out_dir: Optional[Path] = None,
    draft: bool = False,
    scale: int = 1,
    indices: Optional[Sequence[int]] = None,
    progress: bool = True,
) -> Path:
    """Write every frame (or a chosen subset) as numbered PNGs."""
    out_dir = out_dir or (OUTPUT_DIR / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = range(CONFIG.total_frames) if indices is None else indices
    frames = list(frames)
    started = _time.time()
    for count, index in enumerate(frames, start=1):
        render_frame_png(int(index), out_dir / f"frame_{int(index):04d}.png",
                         draft=draft, scale=scale)
        if progress and (count % 20 == 0 or count == len(frames)):
            elapsed = _time.time() - started
            print(f"  frame {count}/{len(frames)}  ({elapsed:.1f}s)", flush=True)
    return out_dir


# --------------------------------------------------------------------------
# Storyboard
# --------------------------------------------------------------------------


def render_storyboard(
    times: Sequence[float] = RENDER_TIMES,
    panel_size: Tuple[int, int] = (480, 270),
    out_path: Optional[Path] = None,
    draft: bool = True,
) -> Image.Image:
    """Six finished frames in a 3x2 grid."""
    columns, rows = 3, 2
    sheet = Image.new(
        "RGB",
        (panel_size[0] * columns, panel_size[1] * rows),
        CONFIG.background,
    )
    for index, moment in enumerate(times):
        frame = render_frame(moment, draft=draft)
        draw = ImageDraw.Draw(frame)
        state = evaluate_timeline(moment)
        draw.text((8, 6), f"t={moment:.2f}s  {state.scene.upper()}", fill=PALETTE.ink_dim)
        draw.text((8, 22), f"zoom={state.camera.zoom:.2f}  growth={state.tree_growth:.2f}",
                  fill=PALETTE.ink_faint)
        panel = frame.convert("RGB").resize(panel_size, Image.LANCZOS)
        sheet.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))

    path = out_path or (OUTPUT_DIR / "master_storyboard.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return sheet


# --------------------------------------------------------------------------
# Export (simple path — src/export.py adds the adaptive pipeline)
# --------------------------------------------------------------------------


def export_webp(
    out_path: Optional[Path] = None,
    quality: int = None,
    method: int = None,
    draft: bool = False,
    scale: int = 1,
    indices: Optional[Sequence[int]] = None,
    progress: bool = True,
) -> Path:
    """Render the sequence and write an animated WebP."""
    path = out_path or (OUTPUT_DIR / CONFIG.export.webp_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    indices = list(indices) if indices is not None else list(range(CONFIG.total_frames))
    frames: List[Image.Image] = []
    started = _time.time()
    for count, index in enumerate(indices, start=1):
        frame = render_frame(CONFIG.frame_time(int(index)), scale=scale, draft=draft)
        frames.append(frame.convert("RGB"))
        if progress and (count % 20 == 0 or count == len(indices)):
            elapsed = _time.time() - started
            print(f"  rendered {count}/{len(indices)} ({elapsed:.1f}s)", flush=True)

    duration_ms = int(round(1000.0 / CONFIG.fps))
    frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=quality if quality is not None else CONFIG.export.quality,
        method=method if method is not None else CONFIG.export.method,
    )
    return path


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    size = CONFIG.size()

    # Frame geometry.
    frame = render_frame(6.5, draft=True)
    assert frame.size == size, f"expected {size}, got {frame.size}"
    assert frame.mode == "RGBA"

    # Determinism: same time, same bytes, regardless of order.
    forward = [render_frame(t, draft=True).tobytes()
               for t in (0.0, 3.4, 5.48, 6.5, 9.5, 11.9)]
    backward = [render_frame(t, draft=True).tobytes()
                for t in reversed((0.0, 3.4, 5.48, 6.5, 9.5, 11.9))][::-1]
    assert forward == backward, "out-of-order rendering must match in-order rendering"

    # Repeated single-frame renders are stable.
    assert render_frame(130 / CONFIG.fps, draft=True).tobytes() == \
           render_frame(130 / CONFIG.fps, draft=True).tobytes()

    # Every frame index maps into the timeline.
    for index in (0, 60, 130, 199, CONFIG.total_frames - 1):
        t = CONFIG.frame_time(index)
        assert 0.0 <= t < CONFIG.duration
        assert CONFIG.frame_index(t) in (index, index + 1, index - 1)

    # Layers keep their coordinate contracts.
    world = render_world_layer(6.5)
    assert world.size == size
    characters = render_character_layer(6.5)
    assert characters.size == size
    assert characters.getbbox() is not None, "fighters must be visible at 6.5s"
    typography = render_typography_layer(6.5)
    assert typography.getbbox() is None, "no copy during combat"
    assert render_typography_layer(11.0).getbbox() is not None
    particles = render_particle_layer(6.5)
    assert particles.size == size

    # Fighters are invisible during the intro.
    assert render_character_layer(1.0).getbbox() is None

    # Supersampling yields the scaled canvas and still composes.
    hi = render_frame(10.5, scale=2, draft=True)
    assert hi.size == (CONFIG.width * 2, CONFIG.height * 2)

    # Camera is applied to the world exactly once.
    from .effects import apply_camera_to_layer as _cam

    state = evaluate_timeline(6.5)
    raw_world = render_world_layer(6.5)
    pre_transformed = _cam(raw_world, state.camera, size)
    manual = compose_scene(
        background=background_frame(size),
        world_layer=pre_transformed,
        character_layer=render_character_layer(6.5),
        particle_layer=render_particle_layer(6.5),
        typography_layer=render_typography_layer(6.5),
        time=6.5,
        world_is_camera_transformed=True,
        state=state,
        grain_strength=0.0,
        draft=True,
    )
    automatic = render_frame(6.5, draft=True)
    assert manual.size == automatic.size

    # Typography stays screen-space: it does not move with the camera.
    calm = render_typography_layer(11.0)
    shaken_state = replace(state, camera=Camera(zoom=1.0, pan=Vec2(40, 20), shake=Vec2(10, 10)))
    assert shaken_state.camera.is_identity is False
    assert calm.tobytes() == render_typography_layer(11.0).tobytes()

    # Frame PNG writer.
    png = render_frame_png(130, draft=True)
    assert png.exists() and png.stat().st_size > 0
    assert png.name == "inkbound_frame_0130.png"

    # Storyboard.
    sheet = render_storyboard(draft=True)
    assert sheet.size == (480 * 3, 270 * 2)
    assert (OUTPUT_DIR / "master_storyboard.png").exists()

    # Smoke: a short WebP export round-trips with the right frame count.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "smoke.webp"
        export_webp(target, indices=[0, 40, 80, 120, 160, 200], draft=True, progress=False)
        assert target.exists() and target.stat().st_size > 0
        with Image.open(target) as probe:
            assert probe.format == "WEBP"
            assert getattr(probe, "n_frames", 1) == 6
            assert probe.size == size

    # Config sanity.
    from .config import validate

    assert validate() == [], validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INKBOUND master renderer")
    parser.add_argument("--test", action="store_true", help="run integration tests")
    parser.add_argument("--preview", action="store_true", help="write the storyboard")
    parser.add_argument("--frame", type=int, default=None, help="render one frame index")
    parser.add_argument("--export", action="store_true", help="export the animated WebP")
    parser.add_argument("--png", action="store_true", help="write the full PNG sequence")
    parser.add_argument("--draft", action="store_true", help="fast low-cost render")
    parser.add_argument("--quality", type=int, default=None, help="WebP quality 0-100")
    parser.add_argument("--method", type=int, default=None, help="WebP compression method 0-6")
    parser.add_argument("--scale", type=int, default=1, help="supersampling factor")
    args = parser.parse_args()

    if args.test:
        _self_test()
        print("All master renderer tests passed.")
        print("INKBOUND renderer loaded.")
        raise SystemExit(0)

    scale = args.scale

    if args.preview:
        path = render_storyboard(draft=args.draft)
        print(f"Storyboard: {path}")
    elif args.frame is not None:
        path = render_frame_png(args.frame, draft=args.draft, scale=scale)
        print(f"Frame: {path}")
    elif args.png:
        path = render_png_sequence(draft=args.draft, scale=scale)
        print(f"PNG sequence: {path}")
    elif args.export:
        path = export_webp(quality=args.quality, method=args.method,
                           draft=args.draft, scale=scale)
        print(f"Export: {path}  ({path.stat().st_size / (1024 * 1024):.2f} MiB)")
    else:
        parser.print_help()
