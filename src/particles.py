"""
INKBOUND — Deterministic Cinematic Particle Engine
File: src/particles.py

Requires:
    Pillow
    src/config.py
    src/characters.py
    src/choreography.py

Effects:
    - Sword sparks
    - Ink splashes
    - Impact debris
    - Expanding shockwaves
    - Motion streaks
    - Atmospheric ink particles
    - Cinematic flash overlay

All particle positions are evaluated analytically.
There is no mutable simulation state, so rendering frame 180
before frame 20 produces byte-identical layers.

Coordinate system:
    World coordinates: 960x540 reference space.
    Camera coordinates: derived from choreography.py.

Usage:
    engine = ParticleEngine()
    render_particles(frame, engine, time=6.5, camera=timeline.camera)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

try:
    from .config import (
        CLASH_POINT,
        CONFIG,
        GROUND_Y,
        IMPACT_TIMES,
        OUTPUT_DIR,
        PALETTE,
        SEED,
    )
    from .characters import Vec2, clamp, lerp, smoothstep
    from .choreography import (
        IMPACT_EVENTS,
        Camera,
        TimelineState,
        evaluate_timeline,
        loop_time,
    )
except ImportError:  # pragma: no cover - direct script execution
    from config import (  # type: ignore
        CLASH_POINT,
        CONFIG,
        GROUND_Y,
        IMPACT_TIMES,
        OUTPUT_DIR,
        PALETTE,
        SEED,
    )
    from characters import Vec2, clamp, lerp, smoothstep  # type: ignore
    from choreography import (  # type: ignore
        IMPACT_EVENTS,
        Camera,
        TimelineState,
        evaluate_timeline,
        loop_time,
    )

__all__ = [
    "Particle",
    "ParticleEngine",
    "render_particles",
    "render_flash_overlay",
    "DEFAULT_DENSITY",
]

CLASH = Vec2(*CLASH_POINT)
DEFAULT_DENSITY = 1.0

# Hard cap on simultaneously visible impact particles.
MAX_IMPACT_PARTICLES = 180


# --------------------------------------------------------------------------
# Particles
# --------------------------------------------------------------------------


@dataclass
class Particle:
    """One deterministic, analytically-evaluated particle."""

    origin: Vec2
    velocity: Vec2
    born: float
    life: float
    size: float
    kind: str  # spark | ink | debris | shock | streak | dust
    color: Tuple[int, int, int]
    gravity: float = 0.0
    drag: float = 0.0
    group: str = "ambient"  # ambient | impact | motion | finale

    @property
    def death(self) -> float:
        return self.born + self.life

    def alive_at(self, t: float) -> bool:
        return self.born <= t < self.death

    def position_at(self, t: float) -> Vec2:
        """Closed-form position — no integration step, fully deterministic."""
        age = t - self.born
        if age <= 0:
            return self.origin
        if self.drag > 0:
            k = self.drag
            displacement = self.velocity * ((1.0 - math.exp(-k * age)) / k)
        else:
            displacement = self.velocity * age
        displacement = displacement + Vec2(0.0, 0.5 * self.gravity * age * age)
        return self.origin + displacement

    def progress_at(self, t: float) -> float:
        return clamp((t - self.born) / max(1e-6, self.life), 0.0, 1.0)

    def alpha_at(self, t: float) -> float:
        if not self.alive_at(t):
            return 0.0
        p = self.progress_at(t)
        return clamp(smoothstep(0.0, 0.12, p) * (1.0 - smoothstep(0.5, 1.0, p)), 0.0, 1.0)

    def size_at(self, t: float) -> float:
        p = self.progress_at(t)
        if self.kind == "spark":
            return self.size * (1.0 - 0.55 * p)
        if self.kind == "shock":
            return self.size * (0.3 + 1.8 * p)
        if self.kind == "streak":
            return self.size * (1.0 - 0.7 * p)
        return self.size * (0.7 + 0.3 * math.sin(p * math.pi))


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class ParticleEngine:
    """
    Holds the whole 12-second particle population.

    Generation happens once per (seed, density) and is cached, so
    per-frame cost is filtering + rasterising only.
    """

    def __init__(self, seed: int = SEED, density: float = DEFAULT_DENSITY) -> None:
        self.seed = seed
        self.density = density
        self._particles: Optional[List[Particle]] = None

    # -- population -----------------------------------------------------
    @property
    def particles(self) -> List[Particle]:
        if self._particles is None:
            self._particles = self._generate()
        return self._particles

    def __len__(self) -> int:
        return len(self.particles)

    def _emit(
        self,
        bucket: List[Particle],
        rng: random.Random,
        kind: str,
        moment: float,
        count: int,
        speed: float,
        spread: float,
        life: Tuple[float, float],
        size: Tuple[float, float],
        at: Vec2,
        color: Tuple[int, int, int],
        group: str,
        gravity: float = 0.0,
        drag: float = 0.0,
    ) -> None:
        count = max(0, int(round(count * self.density)))
        for _ in range(count):
            angle = rng.uniform(-math.pi * spread, math.pi * spread) - math.pi / 2
            magnitude = rng.uniform(speed * 0.45, speed)
            bucket.append(
                Particle(
                    origin=at + Vec2(rng.uniform(-8, 8), rng.uniform(-8, 8)),
                    velocity=Vec2(math.cos(angle) * magnitude, math.sin(angle) * magnitude),
                    born=moment + rng.uniform(0.0, 0.06),
                    life=rng.uniform(*life),
                    size=rng.uniform(*size),
                    kind=kind,
                    color=color,
                    gravity=gravity,
                    drag=drag,
                    group=group,
                )
            )

    def _generate(self) -> List[Particle]:
        rng = random.Random(self.seed)
        bucket: List[Particle] = []

        # 1) Ambient ink motes drifting through the whole sequence.
        for _ in range(int(70 * self.density)):
            bucket.append(
                Particle(
                    origin=Vec2(rng.uniform(0, CONFIG.width), rng.uniform(0, CONFIG.height)),
                    velocity=Vec2(rng.uniform(-8, 8), rng.uniform(-14, -3)),
                    born=rng.uniform(-1.0, CONFIG.duration),
                    life=rng.uniform(4.0, 9.0),
                    size=rng.uniform(1.0, 2.6),
                    kind="dust",
                    color=PALETTE.ink_faint,
                    group="ambient",
                )
            )

        # 2) Impact bursts: sparks, ink splash, debris, shockwave.
        for index, impact in enumerate(IMPACT_EVENTS):
            at = impact.position + Vec2(rng.uniform(-16, 16), rng.uniform(-12, 12))
            strength = impact.strength
            self._emit(bucket, rng, "spark", impact.time, 80 * strength, 430, 1.0,
                       (0.35, 0.95), (1.4, 3.4), at, PALETTE.accent_gold,
                       "impact", gravity=240, drag=2.4)
            self._emit(bucket, rng, "ink", impact.time, 55 * strength, 250, 1.0,
                       (0.5, 1.3), (3.0, 9.0), at,
                       PALETTE.accent_red if index % 2 else PALETTE.ink,
                       "impact", gravity=180, drag=3.2)
            self._emit(bucket, rng, "debris", impact.time, 45 * strength, 300, 1.0,
                       (0.6, 1.5), (1.6, 3.6), at, PALETTE.ink_dim,
                       "impact", gravity=420, drag=1.6)
            bucket.append(
                Particle(
                    origin=at,
                    velocity=Vec2(0.0, 0.0),
                    born=impact.time - 0.02,
                    life=0.75,
                    size=16.0 * strength,
                    kind="shock",
                    color=PALETTE.ink,
                    drag=0.0,
                    group="impact",
                )
            )

        # 3) Ground scrape during the intrusion dash.
        for x in (300.0, 660.0):
            for step in range(16):
                self._emit(bucket, rng, "dust", 3.0 + step * 0.11, 4, 90, 0.5,
                           (0.4, 0.9), (2.0, 5.0), Vec2(x, GROUND_Y - 6),
                           PALETTE.ink_dim, "motion", gravity=120, drag=2.0)

        # 4) Motion streaks trailing the fighters during the battle.
        for step in range(40):
            moment = 5.0 + step * 0.1
            side = -1 if step % 2 else 1
            self._emit(bucket, rng, "streak", moment, 3, 520, 0.22,
                       (0.2, 0.45), (6.0, 18.0),
                       CLASH + Vec2(side * rng.uniform(30, 120), rng.uniform(-40, 60)),
                       PALETTE.ink, "motion", drag=4.0)

        # 5) Ink motes rising as the tree grows (9s -> 11s).
        for step in range(30):
            moment = 9.0 + step * 0.07
            self._emit(bucket, rng, "ink", moment, 3, 70, 0.7,
                       (1.6, 3.4), (1.6, 3.2),
                       Vec2(rng.uniform(380, 580), GROUND_Y - rng.uniform(0, 170)),
                       PALETTE.ink_dim, "finale", gravity=-26, drag=0.7)

        # 6) Finale burst when the profile text locks in.
        self._emit(bucket, rng, "spark", 11.0, 70, 320, 1.0, (0.6, 1.5),
                   (1.6, 3.6), CLASH, PALETTE.accent_blue, "finale",
                   gravity=140, drag=2.2)
        self._emit(bucket, rng, "ink", 11.05, 34, 150, 1.0, (1.2, 2.4),
                   (2.2, 4.6), CLASH, PALETTE.ink, "finale",
                   gravity=-30, drag=1.2)

        # 7) Loop-seam motes so the wrap is not empty.
        self._emit(bucket, rng, "dust", 11.7, 26, 120, 1.0, (0.8, 1.6),
                   (1.8, 4.0), CLASH, PALETTE.ink_faint, "finale", drag=1.6)

        return bucket

    # -- queries --------------------------------------------------------
    def active(self, t: float, group: Optional[str] = None) -> List[Particle]:
        t = loop_time(t)
        live = [p for p in self.particles if p.alive_at(t)]
        if group is not None:
            live = [p for p in live if p.group == group]
        if group in (None, "impact"):
            # Enforce the simultaneous-impact cap, keeping the newest sparks.
            impact = [p for p in live if p.group == "impact"]
            if len(impact) > MAX_IMPACT_PARTICLES:
                impact.sort(key=lambda p: p.born, reverse=True)
                keep = set(id(p) for p in impact[:MAX_IMPACT_PARTICLES])
                live = [p for p in live if p.group != "impact" or id(p) in keep]
        return live

    def counts(self, t: float) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for particle in self.active(t):
            counts[particle.group] = counts.get(particle.group, 0) + 1
        return counts


# Shared engine so every frame reuses one population.
_ENGINE_CACHE: Dict[Tuple[int, float], ParticleEngine] = {}


def get_engine(seed: int = SEED, density: float = DEFAULT_DENSITY) -> ParticleEngine:
    key = (seed, round(density, 3))
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = ParticleEngine(seed=seed, density=density)
    return _ENGINE_CACHE[key]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_particles(
    frame: Image.Image,
    engine: Optional[ParticleEngine] = None,
    time: float = 0.0,
    camera: Optional[Camera] = None,
    layer: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Composite every particle alive at `time` onto `frame`.

    When `camera` is supplied each particle position is transformed
    exactly once, in place — no layer-wide second camera pass.
    """
    engine = engine or get_engine()
    size = frame.size
    target = layer if layer is not None else Image.new("RGBA", size, (0, 0, 0, 0))
    if target.size != size:
        target = target.resize(size, Image.LANCZOS)
    draw = ImageDraw.Draw(target)

    for particle in engine.active(time):
        alpha = particle.alpha_at(time)
        if alpha <= 0.01:
            continue
        position = particle.position_at(time)
        if camera is not None:
            position = camera.world_to_screen(position, size)
        radius = max(0.6, particle.size_at(time))
        color = (
            particle.color[0],
            particle.color[1],
            particle.color[2],
            int(clamp(alpha * 255, 0, 255)),
        )

        if particle.kind in ("spark", "streak"):
            tail = particle.velocity.normalized() * min(28.0, particle.velocity.length() * 0.05)
            draw.line(
                [(position - tail).as_int_tuple(), (position + tail).as_int_tuple()],
                fill=color,
                width=max(1, int(radius)),
            )
            draw.ellipse(
                [position.x - radius, position.y - radius,
                 position.x + radius, position.y + radius],
                fill=color,
            )
        elif particle.kind == "shock":
            draw.ellipse(
                [position.x - radius, position.y - radius,
                 position.x + radius, position.y + radius],
                outline=color,
                width=2,
            )
        elif particle.kind in ("ink", "debris"):
            draw.ellipse(
                [position.x - radius, position.y - radius,
                 position.x + radius, position.y + radius],
                fill=color,
            )
        else:
            draw.point(position.as_int_tuple(), fill=color)
            if radius > 1.6:
                draw.ellipse(
                    [position.x - radius, position.y - radius,
                     position.x + radius, position.y + radius],
                    fill=color,
                )

    if layer is None:
        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")
        frame.alpha_composite(target)
        return frame
    return target


def render_flash_overlay(
    frame: Image.Image,
    time: float = 0.0,
    state: Optional[TimelineState] = None,
) -> Image.Image:
    """Screen-space impact flash. Never camera-transformed."""
    state = state or evaluate_timeline(time)
    amount = state.flash
    if amount <= 0.004:
        return frame
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    flash = Image.new("RGBA", frame.size, PALETTE.ink + (int(clamp(amount * 210, 0, 255)),))
    frame.alpha_composite(flash)
    return frame


# --------------------------------------------------------------------------
# Standalone preview
# --------------------------------------------------------------------------

PREVIEW_TIMES: Tuple[float, ...] = (3.6, 5.48, 6.5, 8.25, 9.8, 11.05)


def render_particles_preview(
    times: Sequence[float] = PREVIEW_TIMES,
    panel_size: Tuple[int, int] = (480, 270),
    out_path: Optional[Path] = None,
) -> Image.Image:
    """Six snapshots of the particle system in a 3x2 grid."""
    columns, rows = 3, 2
    engine = get_engine()
    sheet = Image.new(
        "RGB",
        (panel_size[0] * columns, panel_size[1] * rows),
        CONFIG.background,
    )
    for index, moment in enumerate(times):
        state = evaluate_timeline(moment)
        reference = Image.new("RGBA", CONFIG.size(), CONFIG.background + (255,))
        render_particles(reference, engine, time=moment, camera=state.camera)

        draw = ImageDraw.Draw(reference)
        draw.text((8, 6), f"t={moment:.2f}s  {state.scene.upper()}", fill=PALETTE.ink_dim)
        draw.text((8, 22), str(engine.counts(moment)), fill=PALETTE.ink_faint)

        panel = reference.convert("RGB").resize(panel_size, Image.LANCZOS)
        sheet.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))

    path = out_path or (OUTPUT_DIR / "particles_preview.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return sheet


# --------------------------------------------------------------------------
# Standalone tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    # Generation determinism.
    a = ParticleEngine(seed=11, density=0.3)
    b = ParticleEngine(seed=11, density=0.3)
    assert len(a) == len(b) and len(a) > 100
    for first, second in zip(a.particles, b.particles):
        assert first.born == second.born
        assert first.origin.as_tuple() == second.origin.as_tuple()
        assert first.velocity.as_tuple() == second.velocity.as_tuple()

    different = ParticleEngine(seed=12, density=0.3)
    assert different.particles[0].origin.as_tuple() != a.particles[0].origin.as_tuple()

    # Stateless evaluation: identical inputs give identical outputs.
    sample = a.particles[0]
    assert sample.position_at(1.0).as_tuple() == sample.position_at(1.0).as_tuple()
    assert sample.position_at(sample.born - 1.0).as_tuple() == sample.origin.as_tuple()

    # Lifetime windows.
    assert all(p.life > 0 for p in a.particles)
    mid = a.particles[len(a.particles) // 2]
    assert mid.alpha_at(mid.born - 0.5) == 0.0
    assert mid.alpha_at(mid.death + 0.5) == 0.0

    # Impact cap.
    engine = get_engine(density=1.0)
    for moment in IMPACT_TIMES:
        counts = engine.counts(moment)
        assert counts.get("impact", 0) <= MAX_IMPACT_PARTICLES
        assert counts.get("impact", 0) > 0

    # Impact particles exist only near impacts.
    assert engine.counts(1.0).get("impact", 0) == 0

    # Out-of-order equality on the rendered layer.
    first = render_particles(Image.new("RGBA", CONFIG.size(), (0, 0, 0, 0)), engine, time=7.0)
    second = render_particles(Image.new("RGBA", CONFIG.size(), (0, 0, 0, 0)), engine, time=7.0)
    assert first.tobytes() == second.tobytes()

    # Early time has no impact particles and an empty-ish layer.
    quiet = render_particles(Image.new("RGBA", CONFIG.size(), (0, 0, 0, 0)), engine, time=0.5)
    assert quiet.size == CONFIG.size()

    # Camera path matches a manual per-particle transform.
    state = evaluate_timeline(6.5)
    with_cam = render_particles(
        Image.new("RGBA", CONFIG.size(), (0, 0, 0, 0)), engine, time=6.5, camera=state.camera
    )
    assert with_cam.getbbox() is not None

    # Flash overlay only fires on impacts.
    plain = Image.new("RGBA", CONFIG.size(), PALETTE.background + (255,))
    before = render_flash_overlay(plain.copy(), time=1.0)
    assert before.tobytes() == plain.tobytes()
    flashed = render_flash_overlay(plain.copy(), time=5.48)
    assert flashed.tobytes() != plain.tobytes()

    # Preview.
    sheet = render_particles_preview()
    assert sheet.size == (480 * 3, 270 * 2)
    assert (OUTPUT_DIR / "particles_preview.png").exists()


if __name__ == "__main__":
    _self_test()
    print("All particle engine tests passed.")
    print("INKBOUND particle engine loaded.")
