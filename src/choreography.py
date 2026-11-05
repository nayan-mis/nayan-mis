"""
INKBOUND — Cinematic Fight Choreography
File: src/choreography.py

A deterministic 12-second combat timeline.

Requires:
    src/config.py
    src/characters.py

Includes:
    - Wanderer and Twisted Man keyframes
    - Attacks, dodges, jumps and recoils
    - Timed impact events
    - Camera zoom, pan and shake metadata
    - Typography and tree-growth event cues
    - Stateless frame evaluation
    - Standalone tests and storyboard preview

All timestamps are in seconds.
All positions are in 960x540 reference coordinates.

Rendering frame 180 before frame 20 produces exactly the same
character positions as rendering them in chronological order:
`evaluate_timeline` is a pure function of absolute time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

try:
    from .config import (
        ACTORS,
        CONFIG,
        GROUND_Y,
        IMPACT_TIMES,
        OUTPUT_DIR,
        PALETTE,
        SCENES,
        SCENE_BATTLE_END,
        SCENE_FINALE_END,
        SCENE_GROWTH_END,
        SCENE_INTRO_END,
        SCENE_INTRUSION_END,
        SEED,
        TREE_GROWTH_END,
        TREE_GROWTH_START,
        TWISTED_MAN,
        TYPOGRAPHY_FINALE_START,
        TYPOGRAPHY_INTRO_END,
        WANDERER,
        loop_time,
        scene_at,
    )
    from .characters import (
        POSES,
        Pose,
        Vec2,
        clamp,
        draw_skeleton,
        ease_in_out,
        ease_out_cubic,
        get_pose,
        lerp,
        smoothstep,
    )
except ImportError:  # pragma: no cover - direct script execution
    from config import (  # type: ignore
        ACTORS,
        CONFIG,
        GROUND_Y,
        IMPACT_TIMES,
        OUTPUT_DIR,
        PALETTE,
        SCENES,
        SCENE_BATTLE_END,
        SCENE_FINALE_END,
        SCENE_GROWTH_END,
        SCENE_INTRO_END,
        SCENE_INTRUSION_END,
        SEED,
        TREE_GROWTH_END,
        TREE_GROWTH_START,
        TWISTED_MAN,
        TYPOGRAPHY_FINALE_START,
        TYPOGRAPHY_INTRO_END,
        WANDERER,
        loop_time,
        scene_at,
    )
    from characters import (  # type: ignore
        POSES,
        Pose,
        Vec2,
        clamp,
        draw_skeleton,
        ease_in_out,
        ease_out_cubic,
        get_pose,
        lerp,
        smoothstep,
    )

__all__ = [
    "Camera",
    "CameraEvent",
    "ImpactEvent",
    "Transform",
    "ActorState",
    "TimelineState",
    "evaluate_timeline",
    "apply_camera_to_character",
    "camera_at",
    "fighters_at",
    "loop_time",
    "scene_at",
    "STORYBOARD_TIMES",
]

# Storyboard sampling instants used by the standalone preview.
STORYBOARD_TIMES: Tuple[float, ...] = (3.5, 4.1, 5.48, 6.5, 8.25, 11.5)


# --------------------------------------------------------------------------
# Deterministic noise
# --------------------------------------------------------------------------


def smooth_noise(t: float, frequency: float = 1.0, phase: float = 0.0) -> float:
    """Continuous value noise built from three detuned sines."""
    base = t * frequency + phase
    return (
        math.sin(base * 2.0 * math.pi) * 0.6
        + math.sin(base * 4.13 * math.pi + 1.7) * 0.3
        + math.sin(base * 7.31 * math.pi + 4.2) * 0.1
    )


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


@dataclass
class Camera:
    """Zoom / pan / shake in reference pixels, mapped to screen space."""

    zoom: float = 1.0
    pan: Vec2 = field(default_factory=Vec2)
    shake: Vec2 = field(default_factory=Vec2)
    rotation: float = 0.0  # radians

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.zoom - 1.0) < 1e-9
            and self.pan.length_sq() < 1e-12
            and self.shake.length_sq() < 1e-12
            and abs(self.rotation) < 1e-12
        )

    def world_to_screen(self, point: Vec2, size: Tuple[int, int]) -> Vec2:
        width, height = size
        centre = Vec2(width * 0.5, height * 0.5)
        local = point - centre - self.pan
        if self.rotation:
            local = local.rotated(self.rotation)
        local = local * self.zoom
        return local + centre + self.shake

    def screen_to_world(self, point: Vec2, size: Tuple[int, int]) -> Vec2:
        width, height = size
        centre = Vec2(width * 0.5, height * 0.5)
        local = point - centre - self.shake
        if self.rotation:
            local = local.rotated(-self.rotation)
        local = local / self.zoom if self.zoom else Vec2(0.0, 0.0)
        return local + centre + self.pan

    def matrix(self) -> Tuple[float, float, float, float, float, float]:
        """Pillow-style (a, b, c, d, e, f) affine transform."""
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        return (
            self.zoom * cos_r,
            self.zoom * sin_r,
            -self.zoom * sin_r,
            self.zoom * cos_r,
            0.0,
            0.0,
        )


@dataclass(frozen=True)
class CameraEvent:
    """Metadata describing one deliberate camera cue."""

    time: float
    kind: str  # "zoom" | "shake" | "pan" | "roll"
    value: float
    duration: float = 0.0
    note: str = ""


def apply_camera_to_character(
    target: "Vec2 | Pose",
    camera: Camera,
    size: Tuple[int, int],
) -> "Vec2 | Pose":
    """
    Map a point or a whole pose into screen space.

    Characters use this; their layer must NOT also go through
    effects.apply_camera_to_layer(), or the camera is applied twice.
    """
    if isinstance(target, Pose):
        return Pose(
            target.name,
            {joint: camera.world_to_screen(point, size) for joint, point in target.joints.items()},
        )
    return camera.world_to_screen(target, size)


# --------------------------------------------------------------------------
# Keyframe tracks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """Rigid placement of an actor in reference space."""

    position: Vec2
    rotation: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class Keyframe:
    time: float
    position: Tuple[float, float]
    pose: str


def _sample_track(keys: Sequence[Keyframe], t: float) -> Tuple[Vec2, str]:
    """Piecewise-eased interpolation across a sorted keyframe list."""
    if not keys:
        raise ValueError("keyframe track is empty")
    if t <= keys[0].time:
        return Vec2(*keys[0].position), keys[0].pose
    if t >= keys[-1].time:
        return Vec2(*keys[-1].position), keys[-1].pose
    for index in range(len(keys) - 1):
        start, end = keys[index], keys[index + 1]
        if start.time <= t <= end.time:
            span = end.time - start.time
            local = 0.0 if span <= 1e-9 else (t - start.time) / span
            eased = ease_in_out(local)
            position = Vec2(*start.position).lerp_to(Vec2(*end.position), eased)
            # Hold the outgoing pose for the first 60% of the segment so
            # attacks read as poses rather than a continuous morph.
            pose = start.pose if local < 0.6 else end.pose
            return position, pose
    return Vec2(*keys[-1].position), keys[-1].pose


# Wanderer: enters from frame left, presses the attack, is driven back,
# then steps aside for the finale.
WANDERER_KEYS: Tuple[Keyframe, ...] = (
    Keyframe(0.00, (120.0, 306.0), "idle"),
    Keyframe(3.00, (250.0, 306.0), "idle"),
    Keyframe(3.60, (368.0, 306.0), "lunge"),
    Keyframe(4.10, (300.0, 306.0), "recover"),
    Keyframe(4.70, (404.0, 306.0), "coil"),
    Keyframe(5.48, (452.0, 306.0), "clash"),
    Keyframe(6.50, (414.0, 306.0), "recover"),
    Keyframe(7.40, (472.0, 306.0), "lunge"),
    Keyframe(8.25, (444.0, 306.0), "clash"),
    Keyframe(9.00, (376.0, 306.0), "recover"),
    Keyframe(10.00, (300.0, 306.0), "idle"),
    Keyframe(11.00, (238.0, 306.0), "idle"),
    Keyframe(12.00, (214.0, 306.0), "triumph"),
)


def _mirror_track(
    keys: Sequence[Keyframe], width: float = CONFIG.width
) -> Tuple[Keyframe, ...]:
    """Reflect a track across the canvas centreline for the right fighter."""
    return tuple(
        Keyframe(k.time, (width - k.position[0], k.position[1]), k.pose + "_mirror")
        for k in keys
    )


TWISTED_MAN_KEYS: Tuple[Keyframe, ...] = _mirror_track(WANDERER_KEYS)

# The right fighter is not a pure mirror — it holds ground longer and
# attacks on the off-beat, so two hand-written offsets are layered on.
TWISTED_MAN_OFFSETS: Dict[int, Tuple[float, float]] = {
    4: (-26.0, 0.0),   # hesitates before the first clash
    6: (22.0, 0.0),    # gives ground after 6.5s
    8: (-14.0, 0.0),   # braces at 8.25s
}


def _twisted_track() -> Tuple[Keyframe, ...]:
    mirrored = list(_mirror_track(WANDERER_KEYS))
    adjusted: List[Keyframe] = []
    for index, key in enumerate(mirrored):
        dx, dy = TWISTED_MAN_OFFSETS.get(index, (0.0, 0.0))
        adjusted.append(Keyframe(key.time, (key.position[0] + dx, key.position[1] + dy), key.pose))
    return tuple(adjusted)


TWISTED_MAN_KEYS = _twisted_track()


# --------------------------------------------------------------------------
# Impacts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactEvent:
    time: float
    position: Vec2
    strength: float
    label: str = ""

    def envelope(self, t: float) -> float:
        """0..1 attack-decay envelope evaluated at absolute time t."""
        delta = t - self.time
        if delta < -0.14:
            return 0.0
        if delta < 0.0:
            return smoothstep(-0.14, 0.0, delta)
        return math.exp(-delta * 6.5)

    def flash(self, t: float) -> float:
        delta = t - self.time
        if delta < -0.05:
            return 0.0
        if delta < 0.0:
            return smoothstep(-0.05, 0.0, delta) * 0.5
        return math.exp(-delta * 14.0) if delta <= 0.4 else 0.0


IMPACT_EVENTS: Tuple[ImpactEvent, ...] = tuple(
    ImpactEvent(
        time=moment,
        position=Vec2(480.0, 206.0) if index == 0 else Vec2(470.0 + index * 14.0, 214.0),
        strength=(1.0, 0.85, 0.95)[index % 3],
        label=("CLASH", "RIPOSTE", "FINAL CUT")[index % 3],
    )
    for index, moment in enumerate(IMPACT_TIMES)
)

CAMERA_EVENTS: Tuple[CameraEvent, ...] = (
    CameraEvent(0.0, "zoom", 1.06, 3.0, "settle from push-in"),
    CameraEvent(SCENE_INTRO_END, "shake", 6.0, 0.5, "intrusion kick"),
    CameraEvent(SCENE_INTRUSION_END, "zoom", 1.12, 0.6, "battle push"),
    CameraEvent(5.48, "shake", 16.0, 0.9, "first clash"),
    CameraEvent(6.5, "shake", 13.0, 0.8, "riposte"),
    CameraEvent(8.25, "shake", 15.0, 0.9, "final cut"),
    CameraEvent(SCENE_BATTLE_END, "zoom", 0.96, 2.0, "pull back for the tree"),
    CameraEvent(SCENE_GROWTH_END, "zoom", 1.02, 1.0, "finale settle"),
)


# --------------------------------------------------------------------------
# Camera curve
# --------------------------------------------------------------------------


def camera_at(t: float) -> Camera:
    """Camera curve for absolute time t."""
    t = loop_time(t)
    scene, start, end = scene_at(t)
    local = 0.0 if end <= start else (t - start) / (end - start)

    if scene == "intro":
        zoom = lerp(1.06, 1.0, ease_out_cubic(local))
        pan = Vec2(0.0, 0.0)
        rotation = 0.0
    elif scene == "intrusion":
        zoom = lerp(1.0, 1.1, ease_in_out(local))
        pan = Vec2(lerp(0.0, -16.0, ease_in_out(local)), 0.0)
        rotation = math.radians(0.4) * smooth_noise(t, 0.8, phase=5.0)
    elif scene == "battle":
        zoom = lerp(1.1, 1.13, math.sin(local * math.pi)) + 0.02 * local
        pan = Vec2(
            lerp(-16.0, 14.0, ease_in_out(local)),
            lerp(0.0, -8.0, ease_in_out(local)),
        )
        rotation = math.radians(0.6) * smooth_noise(t, 0.8, phase=5.0)
    elif scene == "growth":
        zoom = lerp(1.13, 0.96, ease_in_out(local))
        pan = Vec2(lerp(14.0, 0.0, ease_in_out(local)), lerp(-8.0, 16.0, ease_in_out(local)))
        rotation = 0.0
    else:  # finale
        zoom = lerp(0.96, 1.02, ease_in_out(local))
        pan = Vec2(0.0, lerp(16.0, 0.0, ease_in_out(local)))
        rotation = 0.0

    # Impact shake — decays exponentially after each impact.
    shake = Vec2(0.0, 0.0)
    for index, impact in enumerate(IMPACT_EVENTS):
        delta = t - impact.time
        if 0.0 <= delta <= 1.0:
            decay = math.exp(-delta * 7.0)
            amplitude = 17.0 * impact.strength * decay
            shake = shake + Vec2(
                smooth_noise(delta * 34.0, phase=index * 3.1) * amplitude,
                smooth_noise(delta * 34.0, phase=index * 3.1 + 11.0) * amplitude * 0.7,
            )

    # Intrusion kick — the fighters arriving on set.
    if SCENE_INTRO_END <= t <= SCENE_INTRO_END + 0.6:
        decay = math.exp(-(t - SCENE_INTRO_END) * 6.0)
        shake = shake + Vec2(
            smooth_noise(t * 30.0, phase=1.0) * 7.0 * decay,
            smooth_noise(t * 30.0, phase=6.0) * 5.0 * decay,
        )

    # Idle handheld drift, tiny so text stays readable.
    shake = shake + Vec2(smooth_noise(t, 0.55, phase=2.0) * 1.5,
                         smooth_noise(t, 0.47, phase=9.0) * 1.0)

    return Camera(zoom=zoom, pan=pan, shake=shake, rotation=rotation)


# --------------------------------------------------------------------------
# Actor state
# --------------------------------------------------------------------------


@dataclass
class ActorState:
    """Everything the renderer needs to draw one fighter."""

    key: str
    label: str
    pose_name: str
    pose: Pose
    transform: Transform
    color: Tuple[int, int, int]
    accent: Tuple[int, int, int]
    facing: int
    opacity: float = 1.0
    visible: bool = True

    @property
    def anchor(self) -> Vec2:
        return self.transform.position

    def screen_pose(self, camera: Camera, size: Tuple[int, int]) -> Pose:
        """Pose with the camera already applied — screen space."""
        posed = apply_camera_to_character(self.pose, camera, size)
        assert isinstance(posed, Pose)
        return posed

    def screen_anchor(self, camera: Camera, size: Tuple[int, int]) -> Vec2:
        anchor = apply_camera_to_character(self.transform.position, camera, size)
        assert isinstance(anchor, Vec2)
        return anchor


def _actor_state(
    config,
    keys: Sequence[Keyframe],
    t: float,
    breath_phase: float,
) -> ActorState:
    position, pose_name = _sample_track(keys, t)

    # Fighters are invisible until the intrusion begins.
    if t < SCENE_INTRO_END:
        opacity = 0.0
    else:
        opacity = clamp(smoothstep(SCENE_INTRO_END, SCENE_INTRO_END + 0.35, t), 0.0, 1.0)

    # Finale: they step apart to clear the centre for the profile text.
    if t >= SCENE_GROWTH_END:
        local = clamp((t - SCENE_GROWTH_END) / max(1e-6, SCENE_FINALE_END - SCENE_GROWTH_END), 0.0, 1.0)
        target = config.home[0] + (120.0 if config.facing > 0 else -120.0)
        position = Vec2(lerp(position.x, target, ease_in_out(local)), position.y)
        opacity = 1.0

    # Subtle breathing so idle frames never look frozen.
    breath = math.sin(t * 2.1 + breath_phase) * 1.4
    position = Vec2(position.x, position.y + breath)

    base_name = pose_name[:-7] if pose_name.endswith("_mirror") else pose_name
    pose = get_pose(base_name)
    if config.facing < 0:
        pose = pose.mirrored()

    return ActorState(
        key=config.key,
        label=config.label,
        pose_name=pose_name,
        pose=pose,
        transform=Transform(position=position, scale=config.scale),
        color=config.color,
        accent=config.accent,
        facing=config.facing,
        opacity=opacity,
        visible=opacity > 0.001,
    )


def fighters_at(t: float) -> Dict[str, ActorState]:
    """Both fighters for absolute time t, in reference coordinates."""
    t = loop_time(t)
    return {
        "wanderer": _actor_state(WANDERER, WANDERER_KEYS, t, 0.0),
        "twisted_man": _actor_state(TWISTED_MAN, TWISTED_MAN_KEYS, t, 2.4),
    }


# --------------------------------------------------------------------------
# Timeline state
# --------------------------------------------------------------------------


@dataclass
class TimelineState:
    time: float
    scene: str
    scene_start: float
    scene_end: float
    local: float
    camera: Camera
    wanderer: ActorState
    twisted_man: ActorState
    active_impacts: List[ImpactEvent]
    tree_growth: float
    typography_phase: str
    typography_progress: float
    typography_disruption: float
    ink: float
    flash: float
    speed_lines: float
    loop_fade: float
    camera_events: List[CameraEvent] = field(default_factory=list)

    # -- convenience ----------------------------------------------------
    @property
    def size(self) -> Tuple[int, int]:
        return CONFIG.size()

    @property
    def frame(self) -> int:
        return CONFIG.frame_index(self.time)

    @property
    def fighters(self) -> Dict[str, ActorState]:
        return {"left": self.wanderer, "right": self.twisted_man}

    def actor(self, key: str) -> ActorState:
        if key in ("wanderer", "left"):
            return self.wanderer
        if key in ("twisted_man", "right"):
            return self.twisted_man
        raise KeyError(key)


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------


def _impact_list(t: float, threshold: float = 0.04) -> List[ImpactEvent]:
    return [e for e in IMPACT_EVENTS if e.envelope(t) >= threshold]


def _flash_curve(t: float) -> float:
    return clamp(max((e.flash(t) for e in IMPACT_EVENTS), default=0.0), 0.0, 1.0)


def _tree_growth_curve(t: float) -> float:
    if t <= TREE_GROWTH_START:
        return 0.0
    if t >= TREE_GROWTH_END:
        return 1.0
    return ease_in_out((t - TREE_GROWTH_START) / (TREE_GROWTH_END - TREE_GROWTH_START))


def _ink_curve(t: float) -> float:
    cuts = (SCENE_INTRO_END, SCENE_INTRUSION_END, SCENE_BATTLE_END,
            SCENE_GROWTH_END, CONFIG.duration, 0.0)
    amount = 0.0
    for cut in cuts:
        distance = abs(t - cut)
        if distance < 0.45:
            amount = max(amount, smoothstep(0.45, 0.0, distance))
    return clamp(amount, 0.0, 1.0)


def _speed_line_curve(t: float) -> float:
    if SCENE_INTRO_END <= t < SCENE_BATTLE_END:
        local = (t - SCENE_INTRO_END) / (SCENE_BATTLE_END - SCENE_INTRO_END)
        return smoothstep(0.0, 0.3, local) * (1.0 - 0.35 * local)
    for impact in IMPACT_EVENTS:
        delta = t - impact.time
        if 0.0 <= delta <= 0.6:
            return math.exp(-delta * 5.0) * 0.9 * impact.strength
    if SCENE_BATTLE_END <= t < SCENE_GROWTH_END:
        return 0.3 * smoothstep(SCENE_BATTLE_END, SCENE_BATTLE_END + 0.7, t) * (
            1.0 - smoothstep(SCENE_GROWTH_END - 0.7, SCENE_GROWTH_END, t)
        )
    return 0.0


def _typography_phase(t: float) -> Tuple[str, float]:
    if t < TYPOGRAPHY_INTRO_END:
        return "intro", clamp(t / 2.2, 0.0, 1.0)
    if t >= TYPOGRAPHY_FINALE_START:
        return "finale", clamp((t - TYPOGRAPHY_FINALE_START) / 1.8, 0.0, 1.0)
    return "none", 0.0


def _typography_disruption(t: float) -> float:
    """How hard impacts shove the text around, 0..1."""
    return clamp(max((e.envelope(t) * 0.9 for e in IMPACT_EVENTS), default=0.0), 0.0, 1.0)


def _loop_fade_curve(t: float) -> float:
    """Black wash across the seam so 12s -> 0s is not a hard cut."""
    fade = 0.0
    if t < 0.5:
        fade = max(fade, 1.0 - smoothstep(0.0, 0.5, t))
    if t > CONFIG.duration - 0.5:
        fade = max(fade, smoothstep(CONFIG.duration - 0.5, CONFIG.duration, t))
    return clamp(fade, 0.0, 1.0)


def evaluate_timeline(t: float) -> TimelineState:
    """Pure function of absolute time — the film's single clock."""
    raw = t
    t = loop_time(t)
    scene, start, end = scene_at(t)
    local = 0.0 if end <= start else clamp((t - start) / (end - start), 0.0, 1.0)
    phase, progress = _typography_phase(t)

    return TimelineState(
        time=raw,
        scene=scene,
        scene_start=start,
        scene_end=end,
        local=local,
        camera=camera_at(t),
        wanderer=fighters_at(t)["wanderer"],
        twisted_man=fighters_at(t)["twisted_man"],
        active_impacts=_impact_list(t),
        tree_growth=_tree_growth_curve(t),
        typography_phase=phase,
        typography_progress=progress,
        typography_disruption=_typography_disruption(t),
        ink=_ink_curve(t),
        flash=_flash_curve(t),
        speed_lines=_speed_line_curve(t),
        loop_fade=_loop_fade_curve(t),
        camera_events=[e for e in CAMERA_EVENTS if e.time <= t <= t + 0.05],
    )


# Backwards-compatible alias used by earlier drafts.
evaluate = evaluate_timeline


# --------------------------------------------------------------------------
# Standalone storyboard preview
# --------------------------------------------------------------------------


def _draw_storyboard_panel(
    image: Image.Image,
    state: TimelineState,
    label: str,
) -> None:
    draw = ImageDraw.Draw(image)
    size = image.size

    # Ground.
    ground = state.camera.world_to_screen(Vec2(0.0, GROUND_Y), size)
    ground_end = state.camera.world_to_screen(Vec2(float(size[0]), GROUND_Y), size)
    draw.line([ground.as_int_tuple(), ground_end.as_int_tuple()],
              fill=PALETTE.ink_faint, width=1)

    for actor in (state.wanderer, state.twisted_man):
        if not actor.visible:
            continue
        pose = actor.screen_pose(state.camera, size)
        anchor = actor.screen_anchor(state.camera, size)
        color = actor.accent if actor.accent != PALETTE.ink else actor.color
        draw_skeleton(
            draw,
            pose,
            anchor,
            color=color,
            scale=actor.transform.scale,
            width=5,
            glow=True,
        )

    # Impact ring.
    for impact in state.active_impacts:
        env = impact.envelope(state.time)
        point = state.camera.world_to_screen(impact.position, size)
        radius = int(10 + 46 * (1.0 - env))
        draw.ellipse(
            [point.x - radius, point.y - radius, point.x + radius, point.y + radius],
            outline=PALETTE.ink,
            width=2,
        )

    draw.text((8, 6), f"{label}  {state.scene.upper()}",
              fill=PALETTE.ink_dim)
    draw.text((8, 22), f"t={state.time:.2f}s zoom={state.camera.zoom:.2f}",
              fill=PALETTE.ink_faint)


def render_storyboard(
    times: Sequence[float] = STORYBOARD_TIMES,
    panel_size: Tuple[int, int] = (480, 270),
    out_path: Optional[Path] = None,
) -> Image.Image:
    """Six sampled frames laid out in a 3x2 grid."""
    columns, rows = 3, 2
    sheet = Image.new(
        "RGB",
        (panel_size[0] * columns, panel_size[1] * rows),
        CONFIG.background,
    )
    for index, moment in enumerate(times):
        state = evaluate_timeline(moment)
        panel = Image.new("RGB", panel_size, CONFIG.background)
        # Render at reference size, then downscale for the sheet.
        reference = Image.new("RGB", CONFIG.size(), CONFIG.background)
        _draw_storyboard_panel(reference, state, f"#{index + 1}")
        panel = reference.resize(panel_size, Image.LANCZOS)
        column = index % columns
        row = index // columns
        sheet.paste(panel, (column * panel_size[0], row * panel_size[1]))

    path = out_path or (OUTPUT_DIR / "choreography_storyboard.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return sheet


# --------------------------------------------------------------------------
# Standalone tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    size = CONFIG.size()

    # Camera identity.
    identity = Camera()
    assert identity.is_identity
    point = Vec2(100, 200)
    mapped = identity.world_to_screen(point, size)
    assert abs(mapped.x - 100) < 1e-9 and abs(mapped.y - 200) < 1e-9

    centre = Camera(zoom=2.0).world_to_screen(Vec2(size[0] / 2, size[1] / 2), size)
    assert abs(centre.x - size[0] / 2) < 1e-9

    cam = camera_at(5.48)
    back = cam.screen_to_world(cam.world_to_screen(point, size), size)
    assert back.distance_to(point) < 1e-6

    # Determinism and out-of-order equality.
    times = [0.0, 1.5, 3.4, 5.48, 6.5, 8.25, 9.9, 11.5]
    forward = [evaluate_timeline(t) for t in times]
    backward = [evaluate_timeline(t) for t in reversed(times)][::-1]
    for a, b in zip(forward, backward):
        assert a.camera.zoom == b.camera.zoom
        assert a.tree_growth == b.tree_growth
        assert a.wanderer.transform.position.as_tuple() == b.wanderer.transform.position.as_tuple()
        assert a.twisted_man.transform.position.as_tuple() == b.twisted_man.transform.position.as_tuple()
        assert [i.time for i in a.active_impacts] == [i.time for i in b.active_impacts]

    # Scene boundaries.
    assert evaluate_timeline(0.0).scene == "intro"
    assert evaluate_timeline(3.5).scene == "intrusion"
    assert evaluate_timeline(6.5).scene == "battle"
    assert evaluate_timeline(9.5).scene == "growth"
    assert evaluate_timeline(11.5).scene == "finale"

    # Loop wrapping.
    assert abs(loop_time(12.5) - 0.5) < 1e-9
    assert abs(loop_time(-0.5) - 11.5) < 1e-9

    # Actors are invisible during the intro, visible afterwards.
    assert evaluate_timeline(1.0).wanderer.opacity == 0.0
    assert evaluate_timeline(1.0).wanderer.visible is False
    assert evaluate_timeline(4.1).wanderer.visible is True
    assert evaluate_timeline(4.1).twisted_man.visible is True

    # Impacts fire on schedule.
    for moment in IMPACT_TIMES:
        state = evaluate_timeline(moment)
        assert state.active_impacts, f"no active impact at {moment}"
        assert any(abs(i.time - moment) < 1e-9 for i in state.active_impacts)
        assert state.flash > 0.5
        assert state.typography_disruption > 0.4

    assert evaluate_timeline(1.0).active_impacts == []
    assert evaluate_timeline(4.0).flash == 0.0

    # Tree growth: zero before 9s, one after 11s, monotonic in between.
    growth = [evaluate_timeline(t).tree_growth for t in (0.0, 8.9, 9.5, 10.0, 10.9, 11.5)]
    assert growth[0] == 0.0 and growth[1] == 0.0 and growth[-1] == 1.0
    assert all(growth[i] <= growth[i + 1] + 1e-9 for i in range(len(growth) - 1))

    # Typography phases.
    assert evaluate_timeline(1.0).typography_phase == "intro"
    assert evaluate_timeline(6.5).typography_phase == "none"
    assert evaluate_timeline(11.5).typography_phase == "finale"

    # Curves stay inside range across the whole timeline.
    for index in range(0, CONFIG.total_frames, 5):
        state = evaluate_timeline(CONFIG.frame_time(index))
        assert 0.0 <= state.flash <= 1.0
        assert 0.0 <= state.ink <= 1.0
        assert 0.0 <= state.loop_fade <= 1.0
        assert 0.0 <= state.speed_lines <= 1.0
        assert 0.9 <= state.camera.zoom <= 1.2
        assert state.wanderer.transform.position.x >= 0.0
        assert state.twisted_man.transform.position.x <= CONFIG.width

    # Keyframe tracks are monotonic in time and mirrored across the canvas.
    assert all(WANDERER_KEYS[i].time < WANDERER_KEYS[i + 1].time for i in range(len(WANDERER_KEYS) - 1))
    # Index 3 carries no offset, so it is a pure mirror across the centreline.
    mirror_w = WANDERER_KEYS[3].position[0]
    mirror_t = TWISTED_MAN_KEYS[3].position[0]
    assert abs((CONFIG.width - mirror_w) - mirror_t) < 1e-6

    # The two fighters never occupy the same point.
    for moment in (3.6, 5.48, 6.5, 8.25, 9.5):
        state = evaluate_timeline(moment)
        gap = state.wanderer.transform.position.distance_to(state.twisted_man.transform.position)
        assert gap > 40.0, f"fighters overlap at {moment}s (gap {gap:.1f})"

    # Camera events exist and are sorted.
    assert len(CAMERA_EVENTS) >= 6
    assert all(CAMERA_EVENTS[i].time <= CAMERA_EVENTS[i + 1].time
               for i in range(len(CAMERA_EVENTS) - 1))

    # Character camera mapping is a pure point transform.
    posed = apply_camera_to_character(POSES["idle"], cam, size)
    assert isinstance(posed, Pose)
    single = apply_camera_to_character(Vec2(330, 300), cam, size)
    assert isinstance(single, Vec2)

    # Storyboard renders.
    sheet = render_storyboard()
    assert sheet.size == (480 * 3, 270 * 2)
    assert (OUTPUT_DIR / "choreography_storyboard.png").exists()


if __name__ == "__main__":
    _self_test()
    print("All choreography tests passed.")
    print("INKBOUND choreography loaded.")
