"""
INKBOUND — Character Geometry and Poses
File: src/characters.py

Vector maths, easing helpers, a named-pose skeleton system and a
Pillow drawer for the two stick fighters.

Poses live in a local space where the fighter's hips are near the
origin and +x points forward.  choreography.py translates, mirrors
and interpolates them; this module never touches the camera.

Requires: Pillow, src/config.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw

try:
    from .config import CONFIG, PALETTE
except ImportError:  # pragma: no cover - direct script execution
    from config import CONFIG, PALETTE  # type: ignore

# --------------------------------------------------------------------------
# Scalar helpers
# --------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into [low, high]."""
    if low > high:
        low, high = high, low
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def inv_lerp(a: float, b: float, value: float) -> float:
    if abs(b - a) < 1e-12:
        return 0.0
    return (value - a) / (b - a)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite smoothstep — 0 below edge0, 1 above edge1, S-curve between."""
    if abs(edge1 - edge0) < 1e-12:
        return 0.0 if x < edge0 else 1.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# Vector
# --------------------------------------------------------------------------


@dataclass
class Vec2:
    """Minimal 2D vector with immutable-style operators."""

    x: float = 0.0
    y: float = 0.0

    # -- construction ----------------------------------------------------
    @classmethod
    def from_tuple(cls, pair: Sequence[float]) -> "Vec2":
        return cls(float(pair[0]), float(pair[1]))

    # -- arithmetic ------------------------------------------------------
    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        if abs(scalar) < 1e-12:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    # -- queries ---------------------------------------------------------
    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec2":
        magnitude = self.length()
        if magnitude < 1e-12:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / magnitude, self.y / magnitude)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def perpendicular(self) -> "Vec2":
        return Vec2(-self.y, self.x)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()

    def rotated(self, radians: float) -> "Vec2":
        c, s = math.cos(radians), math.sin(radians)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    # -- conversion ------------------------------------------------------
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def lerp_to(self, other: "Vec2", t: float) -> "Vec2":
        return Vec2(lerp(self.x, other.x, t), lerp(self.y, other.y, t))


def lerp_vec(a: Vec2, b: Vec2, t: float) -> Vec2:
    return a.lerp_to(b, t)


# --------------------------------------------------------------------------
# Skeleton
# --------------------------------------------------------------------------

JOINT_ORDER: Tuple[str, ...] = (
    "head",
    "neck",
    "hip",
    "shoulder_l",
    "elbow_l",
    "hand_l",
    "shoulder_r",
    "elbow_r",
    "hand_r",
    "knee_l",
    "foot_l",
    "knee_r",
    "foot_r",
)

# Bones connect joints in drawing order (thick core first).
BONES: Tuple[Tuple[str, str], ...] = (
    ("neck", "hip"),
    ("neck", "shoulder_l"),
    ("neck", "shoulder_r"),
    ("shoulder_l", "elbow_l"),
    ("elbow_l", "hand_l"),
    ("shoulder_r", "elbow_r"),
    ("elbow_r", "hand_r"),
    ("hip", "knee_l"),
    ("knee_l", "foot_l"),
    ("hip", "knee_r"),
    ("knee_r", "foot_r"),
)


@dataclass
class Pose:
    """A named set of joint positions in fighter-local space."""

    name: str
    joints: Dict[str, Vec2] = field(default_factory=dict)

    def get(self, joint: str, default: Vec2 | None = None) -> Vec2:
        if joint in self.joints:
            return self.joints[joint]
        if default is not None:
            return default
        raise KeyError(joint)

    def mirrored(self) -> "Pose":
        """Flip left/right limbs and negate x — used for the right fighter."""
        swapped = {}
        for key, value in self.joints.items():
            if key.endswith("_l"):
                new_key = key[:-2] + "_r"
            elif key.endswith("_r"):
                new_key = key[:-2] + "_l"
            else:
                new_key = key
            swapped[new_key] = Vec2(-value.x, value.y)
        return Pose(self.name + "_mirror", swapped)

    def translated(self, offset: Vec2) -> "Pose":
        return Pose(self.name, {k: v + offset for k, v in self.joints.items()})

    def scaled(self, factor: float) -> "Pose":
        return Pose(self.name, {k: v * factor for k, v in self.joints.items()})


def _pose_from_pairs(name: str, pairs: Dict[str, Tuple[float, float]]) -> Pose:
    return Pose(name, {k: Vec2(v[0], v[1]) for k, v in pairs.items()})


# Local space: +x = forward (toward the opponent), +y = down.
# Hip sits near (0, 0); the figure is roughly 150 units tall.
_POSE_DEFINITIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Standing guard, weight centred.
    "idle": {
        "head": (6, -78),
        "neck": (4, -58),
        "hip": (0, 0),
        "shoulder_l": (4, -54),
        "elbow_l": (22, -40),
        "hand_l": (34, -30),
        "shoulder_r": (2, -54),
        "elbow_r": (-16, -42),
        "hand_r": (-26, -28),
        "knee_l": (16, 42),
        "foot_l": (22, 86),
        "knee_r": (-14, 44),
        "foot_r": (-24, 86),
    },
    # Coiled, leaning forward before the dash.
    "coil": {
        "head": (16, -70),
        "neck": (12, -52),
        "hip": (-6, 4),
        "shoulder_l": (12, -48),
        "elbow_l": (34, -36),
        "hand_l": (46, -22),
        "shoulder_r": (8, -48),
        "elbow_r": (-14, -34),
        "hand_r": (-30, -18),
        "knee_l": (20, 40),
        "foot_l": (26, 86),
        "knee_r": (-24, 42),
        "foot_r": (-40, 84),
    },
    # Full extension mid-lunge, sword arm out.
    "lunge": {
        "head": (34, -66),
        "neck": (26, -48),
        "hip": (-2, 2),
        "shoulder_l": (28, -46),
        "elbow_l": (56, -40),
        "hand_l": (78, -34),
        "shoulder_r": (22, -46),
        "elbow_r": (-6, -30),
        "hand_r": (-34, -14),
        "knee_l": (34, 38),
        "foot_l": (56, 78),
        "knee_r": (-32, 40),
        "foot_r": (-62, 82),
    },
    # Clashing — braced against the impact.
    "clash": {
        "head": (22, -72),
        "neck": (16, -52),
        "hip": (-8, 4),
        "shoulder_l": (20, -50),
        "elbow_l": (48, -46),
        "hand_l": (70, -44),
        "shoulder_r": (14, -50),
        "elbow_r": (-12, -38),
        "hand_r": (-36, -24),
        "knee_l": (24, 42),
        "foot_l": (34, 86),
        "knee_r": (-34, 40),
        "foot_r": (-58, 84),
    },
    # Recoiling after the exchange.
    "recover": {
        "head": (-4, -76),
        "neck": (-6, -56),
        "hip": (-10, 2),
        "shoulder_l": (-4, -52),
        "elbow_l": (14, -40),
        "hand_l": (24, -26),
        "shoulder_r": (-8, -52),
        "elbow_r": (-26, -40),
        "hand_r": (-38, -30),
        "knee_l": (10, 42),
        "foot_l": (14, 86),
        "knee_r": (-26, 44),
        "foot_r": (-44, 86),
    },
    # Victory / finale stance, blade raised.
    "triumph": {
        "head": (8, -84),
        "neck": (6, -62),
        "hip": (0, 0),
        "shoulder_l": (8, -58),
        "elbow_l": (30, -60),
        "hand_l": (44, -78),
        "shoulder_r": (2, -58),
        "elbow_r": (-18, -46),
        "hand_r": (-30, -32),
        "knee_l": (14, 44),
        "foot_l": (20, 88),
        "knee_r": (-14, 44),
        "foot_r": (-22, 88),
    },
}

POSES: Dict[str, Pose] = {
    name: _pose_from_pairs(name, pairs) for name, pairs in _POSE_DEFINITIONS.items()
}


def get_pose(name: str) -> Pose:
    try:
        return POSES[name]
    except KeyError as exc:
        raise KeyError(f"unknown pose {name!r}; known: {sorted(POSES)}") from exc


def blend_poses(a: Pose, b: Pose, t: float) -> Pose:
    """Linear interpolation between two poses, joint by joint."""
    t = clamp(t, 0.0, 1.0)
    joints: Dict[str, Vec2] = {}
    for joint in JOINT_ORDER:
        start = a.get(joint, b.get(joint))
        end = b.get(joint, start)
        joints[joint] = start.lerp_to(end, t)
    # Include any extra joints present on either side.
    for joint in set(a.joints) | set(b.joints):
        if joint in joints:
            continue
        start = a.get(joint, b.get(joint))
        end = b.get(joint, start)
        joints[joint] = start.lerp_to(end, t)
    return Pose(f"{a.name}->{b.name}", joints)


def pose_chain(sequence: Sequence[Tuple[str, float]], t: float) -> Pose:
    """
    Walk a list of (pose_name, duration) segments for local time t.

    Time clamps to the last pose, so a chain can describe a whole
    12-second routine without allocation per frame.
    """
    if not sequence:
        raise ValueError("pose_chain requires at least one segment")
    elapsed = 0.0
    for index, (name, span) in enumerate(sequence):
        span = max(0.0, float(span))
        if t <= elapsed + span or index == len(sequence) - 1:
            local = 0.0 if span <= 0 else clamp((t - elapsed) / span, 0.0, 1.0)
            current = get_pose(name)
            if index + 1 >= len(sequence):
                return current
            following = get_pose(sequence[index + 1][0])
            return blend_poses(current, following, ease_in_out(local))
        elapsed += span
    return get_pose(sequence[-1][0])


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    pose: Pose,
    offset: Vec2 | None = None,
    *,
    color: Tuple[int, int, int] = PALETTE.ink,
    head_color: Tuple[int, int, int] | None = None,
    scale: float = 1.0,
    width: int = 5,
    head_radius: float = 12.0,
    glow: bool = False,
) -> None:
    """Draw a pose onto a Pillow image."""
    offset = offset or Vec2(0.0, 0.0)
    head_color = head_color or color

    def place(joint: str, fallback: Tuple[float, float] = (0.0, 0.0)) -> Tuple[int, int]:
        point = pose.get(joint, Vec2(fallback[0], fallback[1]))
        world = offset + point * scale
        return world.as_int_tuple()

    # Bones.
    for start_joint, end_joint in BONES:
        draw.line(
            [place(start_joint), place(end_joint)],
            fill=color,
            width=max(1, int(round(width * scale))),
        )

    # Head.
    hx, hy = place("head")
    radius = max(2, int(round(head_radius * scale)))
    draw.ellipse(
        [hx - radius, hy - radius, hx + radius, hy + radius],
        fill=head_color,
    )

    if glow:
        # Cheap outer ring so the silhouette reads against the tree.
        draw.ellipse(
            [hx - radius - 2, hy - radius - 2, hx + radius + 2, hy + radius + 2],
            outline=color,
            width=1,
        )


def render_fighter_layer(
    size: Tuple[int, int],
    placements: Iterable[Tuple[Pose, Vec2, Vec2, float, Tuple[int, int, int]]],
) -> Image.Image:
    """
    Render several fighters into one RGBA layer.

    placements yields (pose, offset, _unused, scale, color) tuples.
    The unused slot keeps the tuple shape stable for choreography.
    """
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for pose, offset, _anchor, scale, color in placements:
        draw_skeleton(draw, pose, offset, color=color, scale=scale)
    return layer


def skeleton_bounds(pose: Pose, offset: Vec2 | None = None, scale: float = 1.0) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds (left, top, right, bottom) of a posed skeleton."""
    offset = offset or Vec2(0.0, 0.0)
    points = [offset + joint * scale for joint in pose.joints.values()]
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------
# Standalone tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    # Vector maths.
    v = Vec2(3, 4)
    assert abs(v.length() - 5.0) < 1e-9
    assert v.normalized().length() == 1.0
    assert (v + Vec2(1, 1)).as_tuple() == (4, 5)
    assert (v * 2).as_tuple() == (6, 8)

    # Easing.
    assert clamp(5, 0, 1) == 1
    assert clamp(-5, 0, 1) == 0
    assert smoothstep(0, 1, -1) == 0.0
    assert smoothstep(0, 1, 2) == 1.0
    assert abs(smoothstep(0, 1, 0.5) - 0.5) < 1e-9

    # Poses.
    assert set(POSES) >= {"idle", "coil", "lunge", "clash", "recover", "triumph"}
    for pose in POSES.values():
        for joint in JOINT_ORDER:
            assert joint in pose.joints, f"{pose.name} missing {joint}"

    # Interpolation stays inside the joint set.
    mid = blend_poses(POSES["idle"], POSES["lunge"], 0.5)
    assert set(mid.joints) == set(POSES["idle"].joints)
    assert mid.get("hand_r").x != POSES["idle"].get("hand_r").x

    # Mirroring swaps limbs.
    mirrored = POSES["idle"].mirrored()
    assert mirrored.get("hand_l").x == -POSES["idle"].get("hand_r").x

    # Chain clamps at both ends.
    chain = [("idle", 1.0), ("lunge", 1.0), ("clash", 1.0)]
    assert pose_chain(chain, -5.0).name.startswith("idle")
    assert pose_chain(chain, 99.0).name.startswith("clash")

    # Rendering produces non-empty pixels.
    img = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_skeleton(draw, POSES["idle"], Vec2(120, 120))
    assert img.getbbox() is not None

    bounds = skeleton_bounds(POSES["lunge"])
    assert bounds[2] > bounds[0] and bounds[3] > bounds[1]


if __name__ == "__main__":
    _self_test()
    print("All character tests passed.")
    print("INKBOUND characters loaded.")
