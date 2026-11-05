"""
INKBOUND — Procedural World Tree
File: src/tree.py

Deterministic curved branching and growth animation.

Features:
    - Recursive binary branching
    - Organic curved geometry
    - Tapered trunk and branches
    - Procedural underground roots
    - Deterministic leaf placement
    - Falling leaves
    - Smooth growth animation
    - Stateless rendering
    - Standalone tests and preview

Dependencies:
    Pillow
    src/config.py
    src/characters.py
    src/choreography.py   (growth timeline only)

All geometry is generated in 960x540 reference coordinates,
using TREE settings from config.py and the 9s-11s growth
window defined there.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

try:
    from .config import CONFIG, OUTPUT_DIR, PALETTE, TREE, TREE_GROWTH_END, TREE_GROWTH_START
    from .characters import Vec2, clamp, lerp, smoothstep
    from .choreography import camera_at, evaluate_timeline
except ImportError:  # pragma: no cover - direct script execution
    from config import CONFIG, OUTPUT_DIR, PALETTE, TREE, TREE_GROWTH_END, TREE_GROWTH_START  # type: ignore
    from characters import Vec2, clamp, lerp, smoothstep  # type: ignore
    from choreography import camera_at, evaluate_timeline  # type: ignore

__all__ = [
    "Branch",
    "Leaf",
    "ProceduralTree",
    "build_tree",
    "get_tree",
    "render_tree",
    "render_tree_at_time",
    "visible_branches",
    "tree_bounds",
    "growth_at",
]

ROOT_ORIGIN = Vec2(*TREE.origin)


# --------------------------------------------------------------------------
# Growth curve — mirrors choreography's tree_growth exactly
# --------------------------------------------------------------------------


def growth_at(t: float) -> float:
    """0..1 growth scalar for absolute time t (stateless)."""
    if t <= TREE_GROWTH_START:
        return 0.0
    if t >= TREE_GROWTH_END:
        return 1.0
    return smoothstep(TREE_GROWTH_START, TREE_GROWTH_END, t)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass
class Branch:
    """One tapered segment of the tree."""

    start: Vec2
    end: Vec2
    width: float
    depth: int
    birth: float  # 0..1 growth value at which this branch appears
    children: List["Branch"] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    def iter_depth_first(self):
        yield self
        for child in self.children:
            yield from child.iter_depth_first()


@dataclass(frozen=True)
class Leaf:
    """A single leaf anchored to a branch tip."""

    position: Vec2
    size: float
    birth: float
    phase: float
    color: Tuple[int, int, int]


def build_tree(
    seed: int = TREE.seed,
    max_depth: int = TREE.max_depth,
    trunk_length: float = TREE.trunk_length,
    root_origin: Vec2 = None,
) -> Branch:
    """
    Deterministically grow an ink tree rooted at `root_origin`.

    The tree grows upward from the ground line; branches split
    with seeded angle/length jitter so two seeds never match.
    """
    root_origin = root_origin or ROOT_ORIGIN
    rng = random.Random(seed)

    def grow(start: Vec2, angle: float, length: float, width: float, depth: int, birth: float) -> Branch:
        curve = rng.uniform(-0.18, 0.18)
        end_angle = angle + curve
        end = start + Vec2(math.cos(end_angle) * length, math.sin(end_angle) * length)
        branch = Branch(start=start, end=end, width=width, depth=depth, birth=birth)

        if depth >= max_depth or length < 8.0:
            return branch

        children_count = 2 if depth < max_depth - 1 else rng.choice((1, 2))
        next_birth = min(1.0, birth + 0.10 + rng.uniform(0.0, 0.06))
        for index in range(children_count):
            if children_count == 1:
                direction = rng.choice((-1, 1))
                spread = rng.uniform(0.15, 0.5)
            else:
                direction = -1 if index % 2 == 0 else 1
                spread = rng.uniform(0.32, 0.72)
            child_angle = end_angle + direction * spread
            child_length = length * rng.uniform(0.62, 0.82)
            child_width = max(1.0, width * rng.uniform(0.55, 0.72))
            branch.children.append(
                grow(end, child_angle, child_length, child_width, depth + 1, next_birth)
            )
        return branch

    trunk_angle = math.radians(-90.0 + rng.uniform(-7.0, 7.0))
    trunk = grow(root_origin, trunk_angle, trunk_length, width=13.0, depth=0, birth=0.0)

    root_group = Branch(start=root_origin, end=root_origin, width=0.0, depth=-1, birth=0.0)
    for _ in range(TREE.root_spokes):
        angle = math.radians(rng.uniform(150.0, 390.0))
        length = rng.uniform(45.0, 95.0)
        root_group.children.append(
            grow(root_origin, angle, length, width=7.0, depth=2, birth=rng.uniform(0.0, 0.18))
        )

    wrapper = Branch(start=root_origin, end=root_origin, width=0.0, depth=-1, birth=0.0)
    wrapper.children = [trunk, root_group]
    return wrapper


_TREE_CACHE: dict = {}


def get_tree(seed: int = TREE.seed) -> Branch:
    if seed not in _TREE_CACHE:
        _TREE_CACHE[seed] = build_tree(seed=seed)
    return _TREE_CACHE[seed]


def visible_branches(tree: Branch, growth: float) -> List[Branch]:
    """Branches whose birth value is reached by the current growth scalar."""
    growth = clamp(growth, 0.0, 1.0)
    if growth <= 0.0:
        return []
    return [b for b in tree.iter_depth_first() if b.birth <= growth and b.length > 0.01]


def branch_tip_progress(branch: Branch, growth: float) -> float:
    """How far along this branch the sap has reached, 0..1."""
    if growth <= branch.birth:
        return 0.0
    return clamp((growth - branch.birth) / 0.14, 0.0, 1.0)


def build_leaves(tree: Branch, seed: int = TREE.seed, count: int = None) -> List[Leaf]:
    """Deterministic leaf cloud attached to the outer branches."""
    count = count if count is not None else TREE.leaf_count
    rng = random.Random(seed ^ 0x5EED)
    tips = [b for b in tree.iter_depth_first() if b.depth >= TREE.max_depth - 2 and b.length > 0.01]
    if not tips:
        return []
    leaves: List[Leaf] = []
    for index in range(count):
        branch = tips[index % len(tips)]
        along = rng.uniform(0.35, 1.0)
        position = branch.start.lerp_to(branch.end, along) + Vec2(
            rng.uniform(-9, 9), rng.uniform(-9, 9)
        )
        leaves.append(
            Leaf(
                position=position,
                size=rng.uniform(*TREE.leaf_size),
                birth=clamp(branch.birth + rng.uniform(0.02, 0.20), 0.0, 1.0),
                phase=rng.uniform(0.0, math.tau),
                color=rng.choice((PALETTE.ink, PALETTE.ink_dim, PALETTE.accent_blue)),
            )
        )
    return leaves


_LEAF_CACHE: dict = {}


def get_leaves(seed: int = TREE.seed) -> List[Leaf]:
    if seed not in _LEAF_CACHE:
        _LEAF_CACHE[seed] = build_leaves(get_tree(seed), seed)
    return _LEAF_CACHE[seed]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_tree(
    size: Tuple[int, int] = None,
    growth: float = 1.0,
    seed: int = TREE.seed,
    color: Tuple[int, int, int] = None,
    root: Optional[Vec2] = None,
    t: float = None,
    with_leaves: bool = True,
) -> Image.Image:
    """
    Rasterise the tree into an RGBA layer the size of `size`.

    Branches are drawn as tapered polygons so the silhouette reads
    as ink rather than a wireframe.  When `t` is supplied, falling
    leaves are animated analytically from that absolute time.
    """
    size = size or CONFIG.size()
    color = color or PALETTE.ink
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    if growth <= 0.0:
        return image

    draw = ImageDraw.Draw(image)
    tree = get_tree(seed)
    branches = visible_branches(tree, growth)
    if not branches:
        return image

    offset = Vec2(0.0, 0.0)
    if root is not None:
        offset = root - ROOT_ORIGIN

    def to_screen(point: Vec2) -> Tuple[float, float]:
        return (point.x + offset.x, point.y + offset.y)

    # Deepest first so thick trunks overlay fine twigs cleanly.
    for branch in sorted(branches, key=lambda b: -b.depth):
        progress = branch_tip_progress(branch, growth)
        if progress <= 0.0:
            continue
        end = branch.start.lerp_to(branch.end, progress)
        alpha = int(clamp(255 * (0.55 + 0.45 * (1.0 - branch.depth / 8.0)), 40, 255))
        fill = (color[0], color[1], color[2], alpha)
        _stroke_tapered(
            draw,
            to_screen(branch.start),
            to_screen(end),
            max(0.7, branch.width * 0.5),
            max(0.4, branch.width * 0.22),
            fill,
        )

    # Ink blot where the roots break the ground.
    radius = lerp(4.0, 26.0, smoothstep(0.05, 0.55, growth))
    cx, cy = to_screen(ROOT_ORIGIN)
    bleed = int(clamp(90 * growth, 0, 160))
    draw.ellipse(
        [cx - radius, cy - radius * 0.4, cx + radius, cy + radius * 0.4],
        fill=(color[0], color[1], color[2], bleed),
    )

    if with_leaves:
        _draw_leaves(draw, growth, offset, t)

    return image


def _draw_leaves(
    draw: ImageDraw.ImageDraw,
    growth: float,
    offset: Vec2,
    t: Optional[float],
) -> None:
    """Leaves appear with growth; when t is given they also drift and fall."""
    for leaf in get_leaves():
        if leaf.birth > growth:
            continue
        reveal = smoothstep(leaf.birth, min(1.0, leaf.birth + 0.12), growth)
        position = leaf.position + offset

        if t is not None:
            # Deterministic sway plus a late-sequence fall.
            sway_x = math.sin(t * 1.7 + leaf.phase) * 3.0
            sway_y = math.cos(t * 1.3 + leaf.phase) * 2.0
            fall = max(0.0, t - (TREE_GROWTH_START + 1.2 + (leaf.phase % 1.0) * 0.8))
            if fall > 0.0:
                position = Vec2(
                    position.x + sway_x + math.sin(fall * 3.0 + leaf.phase) * 14.0,
                    position.y + sway_y + 46.0 * fall * fall,
                )
            else:
                position = Vec2(position.x + sway_x, position.y + sway_y)

        radius = leaf.size * reveal
        if radius < 0.6:
            continue
        alpha = int(clamp(210 * reveal, 0, 255))
        fill = (leaf.color[0], leaf.color[1], leaf.color[2], alpha)
        draw.ellipse(
            [position.x - radius, position.y - radius * 0.7,
             position.x + radius, position.y + radius * 0.7],
            fill=fill,
        )


def _stroke_tapered(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    start_width: float,
    end_width: float,
    fill: Tuple[int, int, int, int],
) -> None:
    """Draw a quadrilateral that tapers from start_width to end_width."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    nx, ny = -dy / length, dx / length
    draw.polygon(
        [
            (sx + nx * start_width, sy + ny * start_width),
            (sx - nx * start_width, sy - ny * start_width),
            (ex - nx * end_width, ey - ny * end_width),
            (ex + nx * end_width, ey + ny * end_width),
        ],
        fill=fill,
    )


class ProceduralTree:
    """Reusable, cached tree that can be stamped onto a frame."""

    def __init__(self, seed: int = TREE.seed, size: Tuple[int, int] = None) -> None:
        self.seed = seed
        self.size = size or CONFIG.size()
        self._geometry = get_tree(seed)
        self._leaves = get_leaves(seed)
        self._cache: dict = {}

    # -- geometry -------------------------------------------------------
    @property
    def geometry(self) -> Branch:
        return self._geometry

    @property
    def leaves(self) -> List[Leaf]:
        return self._leaves

    def branches_at(self, growth: float) -> List[Branch]:
        return visible_branches(self._geometry, growth)

    def bounds(self, growth: float = 1.0) -> Tuple[float, float, float, float]:
        return tree_bounds(growth, self.seed)

    # -- rendering ------------------------------------------------------
    def render(
        self,
        growth: float = 1.0,
        t: Optional[float] = None,
        camera=None,
        color: Tuple[int, int, int] = None,
    ) -> Image.Image:
        """
        Render the tree layer.

        When `camera` is supplied the layer is camera-transformed
        here, so the caller must not transform it a second time.
        """
        layer = render_tree(
            self.size,
            growth=growth,
            seed=self.seed,
            color=color,
            t=t,
        )
        if camera is not None:
            try:
                from .effects import apply_camera_to_layer
            except ImportError:  # pragma: no cover - direct script execution
                from effects import apply_camera_to_layer  # type: ignore

            layer = apply_camera_to_layer(layer, camera, self.size)
        return layer


def render_tree_at_time(
    frame: Image.Image,
    tree: Optional[ProceduralTree] = None,
    time: float = 10.5,
    camera=None,
    alpha: int = 255,
) -> Image.Image:
    """
    Composite the tree onto `frame` at absolute `time`.

    Growth comes from the choreography engine, so the tree stays
    locked to the 9s-11s window without carrying its own clock.

    If `camera` is given, the tree is transformed exactly once
    inside this call.
    """
    tree = tree or ProceduralTree()
    state = evaluate_timeline(time)
    growth = state.tree_growth
    if growth <= 0.0:
        return frame

    layer = tree.render(growth=growth, t=time, camera=camera)
    if alpha < 255:
        alpha_channel = layer.getchannel("A").point(lambda v: int(v * alpha / 255))
        layer.putalpha(alpha_channel)

    if frame.size != layer.size:
        layer = layer.resize(frame.size, Image.LANCZOS)

    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    frame.alpha_composite(layer)
    return frame


def tree_bounds(growth: float = 1.0, seed: int = TREE.seed) -> Tuple[float, float, float, float]:
    """World-space bounding box of the visible tree."""
    branches = visible_branches(get_tree(seed), growth)
    if not branches:
        return (0.0, 0.0, 0.0, 0.0)
    xs: List[float] = []
    ys: List[float] = []
    for branch in branches:
        xs.extend((branch.start.x, branch.end.x))
        ys.extend((branch.start.y, branch.end.y))
    return (min(xs), min(ys), max(xs), max(ys))


def render_roots(
    size: Tuple[int, int] = None,
    growth: float = 1.0,
    seed: int = TREE.seed,
    color: Tuple[int, int, int] = None,
) -> Image.Image:
    """Convenience wrapper: only the sub-ground root structure."""
    return render_tree(size, growth, seed, color, with_leaves=False)


# --------------------------------------------------------------------------
# Standalone preview
# --------------------------------------------------------------------------

PREVIEW_GROWTH: Tuple[float, ...] = (0.05, 0.2, 0.4, 0.6, 0.8, 1.0)


def render_growth_preview(
    stages: Sequence[float] = PREVIEW_GROWTH,
    panel_size: Tuple[int, int] = (480, 270),
    out_path: Optional[Path] = None,
) -> Image.Image:
    """Six growth stages from 5% to 100% laid out in a 3x2 grid."""
    columns, rows = 3, 2
    sheet = Image.new(
        "RGB",
        (panel_size[0] * columns, panel_size[1] * rows),
        CONFIG.background,
    )
    for index, growth in enumerate(stages):
        reference = Image.new("RGB", CONFIG.size(), CONFIG.background)
        layer = render_tree(growth=growth, t=TREE_GROWTH_START + growth * 2.0)
        reference.paste(layer, (0, 0), layer)

        draw = ImageDraw.Draw(reference)
        draw.text((8, 6), f"GROWTH {int(growth * 100)}%", fill=PALETTE.ink_dim)
        draw.text((8, 22), f"t={TREE_GROWTH_START + growth * 2.0:.2f}s", fill=PALETTE.ink_faint)

        panel = reference.resize(panel_size, Image.LANCZOS)
        sheet.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))

    path = out_path or (OUTPUT_DIR / "tree_growth_preview.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return sheet


# --------------------------------------------------------------------------
# Standalone tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    # Seeded determinism.
    a = build_tree(seed=42)
    b = build_tree(seed=42)
    c = build_tree(seed=7)
    a_nodes = [(n.start.as_tuple(), n.end.as_tuple(), n.width) for n in a.iter_depth_first()]
    b_nodes = [(n.start.as_tuple(), n.end.as_tuple(), n.width) for n in b.iter_depth_first()]
    c_nodes = [(n.start.as_tuple(), n.end.as_tuple(), n.width) for n in c.iter_depth_first()]
    assert a_nodes == b_nodes, "same seed must produce identical geometry"
    assert a_nodes != c_nodes, "different seeds must produce different geometry"

    all_nodes = list(a.iter_depth_first())
    assert len(all_nodes) > 40, "tree should be reasonably detailed"
    assert max({n.depth for n in all_nodes}) >= 6

    # Growth window: nothing before 9s, everything after 11s.
    assert growth_at(0.0) == 0.0
    assert growth_at(8.99) == 0.0
    assert growth_at(11.0) == 1.0
    assert growth_at(12.0) == 1.0
    ramp = [growth_at(t) for t in (9.0, 9.5, 10.0, 10.5, 11.0)]
    assert all(ramp[i] <= ramp[i + 1] + 1e-9 for i in range(len(ramp) - 1))
    assert ramp[0] == 0.0 and ramp[-1] == 1.0

    # Matches the choreography engine's curve.
    for t in (9.0, 9.7, 10.4, 11.0):
        assert abs(growth_at(t) - evaluate_timeline(t).tree_growth) < 1e-9

    # Growth gating.
    assert visible_branches(a, 0.0) == []
    early = visible_branches(a, 0.3)
    full = visible_branches(a, 1.0)
    assert len(early) < len(full)
    zero_length = sum(1 for n in all_nodes if n.length <= 0.01)
    assert len(full) == len(all_nodes) - zero_length

    # Bounds move upward as the tree grows (negative y is up).
    small = tree_bounds(0.3)
    big = tree_bounds(1.0)
    assert big[1] <= small[1] + 1e-6

    # Leaves are deterministic and gated by growth.
    leaves_a = build_leaves(a, seed=42)
    leaves_b = build_leaves(a, seed=42)
    assert leaves_a == leaves_b
    assert len(leaves_a) == TREE.leaf_count
    assert sum(1 for leaf in leaves_a if leaf.birth <= 0.3) < len(leaves_a)

    # Rendering.
    zero = render_tree(growth=0.0)
    assert zero.getbbox() is None, "no growth means no pixels"
    full_layer = render_tree(growth=1.0)
    assert full_layer.size == CONFIG.size()
    assert full_layer.getbbox() is not None
    assert render_tree(growth=0.4).getbbox() is not None
    assert render_tree(growth=1.0, with_leaves=False).getbbox() is not None

    # Cache returns the same object.
    assert get_tree(42) is get_tree(42)

    # ProceduralTree + render_tree_at_time integration.
    tree = ProceduralTree()
    assert tree.geometry is get_tree(TREE.seed)
    assert len(tree.branches_at(1.0)) == len(full)
    assert len(tree.leaves) == TREE.leaf_count

    early_frame = Image.new("RGBA", CONFIG.size(), CONFIG.background + (255,))
    untouched = render_tree_at_time(early_frame, tree, time=1.0)
    assert untouched.tobytes() == early_frame.tobytes(), "no growth means no change"

    grown_frame = Image.new("RGBA", CONFIG.size(), CONFIG.background + (255,))
    before = grown_frame.tobytes()
    after = render_tree_at_time(grown_frame, tree, time=10.5).tobytes()
    assert after != before, "growth at 10.5s must add pixels"

    # Camera path also works.
    cam_frame = Image.new("RGBA", CONFIG.size(), CONFIG.background + (255,))
    render_tree_at_time(cam_frame, tree, time=10.5, camera=camera_at(10.5))

    # Preview.
    sheet = render_growth_preview()
    assert sheet.size == (480 * 3, 270 * 2)
    assert (OUTPUT_DIR / "tree_growth_preview.png").exists()


if __name__ == "__main__":
    _self_test()
    print("All procedural tree tests passed.")
    print("INKBOUND tree engine loaded.")
