"""
INKBOUND — Cinematic Typography Engine
File: src/typography.py

Features:
    - Adaptive font loading and fitting
    - GitHub profile typography
    - Intro and finale compositions
    - Animated text reveals
    - Deterministic ink fragmentation
    - Collision-safe text regions
    - Standalone previews and tests

Requires:
    Pillow
    src/config.py
    src/characters.py
    src/choreography.py

Coordinates are based on a 960x540 design canvas.
Rendering supports other output sizes by scaling.

The engine only reports where text lives: `text_safe_zones()`
returns the rectangles the composition renderer must keep clear
of tree branches and fighters.  Typography never clips other
layers itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from .config import CONFIG, OUTPUT_DIR, PALETTE, PROFILE
    from .characters import Vec2, clamp, ease_out_cubic, lerp, smoothstep
    from .choreography import (
        SCENE_FINALE_END,
        TYPOGRAPHY_FINALE_START,
        TYPOGRAPHY_INTRO_END,
        TimelineState,
        evaluate_timeline,
        loop_time,
        smooth_noise,
    )
except ImportError:  # pragma: no cover - direct script execution
    from config import CONFIG, OUTPUT_DIR, PALETTE, PROFILE  # type: ignore
    from characters import Vec2, clamp, ease_out_cubic, lerp, smoothstep  # type: ignore
    from choreography import (  # type: ignore
        SCENE_FINALE_END,
        TYPOGRAPHY_FINALE_START,
        TYPOGRAPHY_INTRO_END,
        TimelineState,
        evaluate_timeline,
        loop_time,
        smooth_noise,
    )

__all__ = [
    "FontBook",
    "TextLine",
    "TextBlock",
    "SafeZone",
    "load_font",
    "fit_text",
    "text_safe_zones",
    "layout_intro",
    "layout_finale",
    "render_typography",
    "render_typography_at_time",
    "PROFILE",
]

Alignment = str  # "left" | "center" | "right"

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------

_FONT_DIRS: Tuple[Path, ...] = tuple(
    p
    for p in (
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    )
    if p.exists()
)

_SANS_CANDIDATES: Tuple[str, ...] = (
    "arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf", "segoeui.ttf",
    "calibrib.ttf", "verdanab.ttf", "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf", "arial.ttf",
)
_SERIF_CANDIDATES: Tuple[str, ...] = (
    "timesbd.ttf", "Times New Roman Bold.ttf", "Georgia Bold.ttf",
    "DejaVuSerif-Bold.ttf", "times.ttf",
)
_MONO_CANDIDATES: Tuple[str, ...] = (
    "consolab.ttf", "consola.ttf", "courbd.ttf", "cour.ttf",
    "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf",
)

_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "sans": _SANS_CANDIDATES,
    "serif": _SERIF_CANDIDATES,
    "mono": _MONO_CANDIDATES,
}

_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _resolve(name: str) -> Optional[Path]:
    for directory in _FONT_DIRS:
        candidate = directory / name
        if candidate.exists():
            return candidate
        # Some distributions nest fonts one level deeper.
        for sub in directory.glob("*/" + name):
            return sub
    return None


def load_font(size: int = 24, family: str = "sans") -> ImageFont.FreeTypeFont:
    """
    Load a font by family with graceful fallback.

    Result is cached per (family, size).  If no font file exists on
    this machine Pillow's built-in bitmap font is used, which keeps
    every test runnable in a minimal environment.
    """
    size = max(6, int(size))
    key = (family, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    chosen: Optional[ImageFont.FreeTypeFont] = None
    for name in _FAMILIES.get(family, _SANS_CANDIDATES):
        path = _resolve(name)
        if path is None:
            continue
        try:
            chosen = ImageFont.truetype(str(path), size)
            break
        except OSError:
            continue

    if chosen is None:
        try:
            chosen = ImageFont.load_default(size=size)  # type: ignore[call-arg]
        except TypeError:
            chosen = ImageFont.load_default()

    _FONT_CACHE[key] = chosen
    return chosen


def text_width(text: str, font: ImageFont.FreeTypeFont) -> float:
    try:
        return float(font.getlength(text))
    except Exception:  # pragma: no cover - bitmap fallback
        return float(len(text)) * 6.0


def fit_text(
    text: str,
    max_width: int,
    family: str = "sans",
    start_size: int = 64,
    min_size: int = 8,
) -> Tuple[ImageFont.FreeTypeFont, int]:
    """Shrink until `text` fits `max_width`. Returns (font, size)."""
    size = max(min_size, int(start_size))
    while size > min_size:
        font = load_font(size, family)
        if text_width(text, font) <= max_width:
            return font, size
        size -= 1
    font = load_font(min_size, family)
    return font, min_size


# --------------------------------------------------------------------------
# Layout primitives
# --------------------------------------------------------------------------


@dataclass
class TextLine:
    text: str
    position: Vec2
    font: ImageFont.FreeTypeFont
    size: int
    color: Tuple[int, int, int] = PALETTE.ink
    alignment: Alignment = "left"
    alpha: float = 1.0
    tracking: float = 0.0  # extra px between characters
    family: str = "sans"
    reveal: float = 1.0  # 0..1 left-to-right reveal
    offset: Vec2 = field(default_factory=Vec2)
    weight: float = 1.0  # opacity multiplier applied on top of alpha

    @property
    def width(self) -> float:
        base = text_width(self.text, self.font)
        if self.tracking and len(self.text) > 1:
            base += self.tracking * (len(self.text) - 1)
        return base

    @property
    def height(self) -> float:
        try:
            box = self.font.getbbox("Ag")
            return float(box[3] - box[1])
        except Exception:  # pragma: no cover
            return float(self.size)


@dataclass
class TextBlock:
    lines: List[TextLine] = field(default_factory=list)
    phase: str = "none"
    progress: float = 0.0

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        if not self.lines:
            return (0, 0, 0, 0)
        left = min(line.position.x for line in self.lines)
        top = min(line.position.y for line in self.lines)
        right = max(line.position.x + line.width for line in self.lines)
        bottom = max(line.position.y + line.height for line in self.lines)
        return (int(left), int(top), int(right), int(bottom))


@dataclass(frozen=True)
class SafeZone:
    """A rectangle the composition renderer must keep clear."""

    rect: Tuple[int, int, int, int]
    phase: str
    strength: float

    @property
    def x0(self) -> int:
        return self.rect[0]

    @property
    def y0(self) -> int:
        return self.rect[1]

    @property
    def x1(self) -> int:
        return self.rect[2]

    @property
    def y1(self) -> int:
        return self.rect[3]

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.rect

    def overlaps(self, other: "SafeZone | Tuple[int, int, int, int]") -> bool:
        rect = other.rect if isinstance(other, SafeZone) else other
        return not (
            rect[2] <= self.rect[0]
            or rect[0] >= self.rect[2]
            or rect[3] <= self.rect[1]
            or rect[1] >= self.rect[3]
        )


# --------------------------------------------------------------------------
# Reveal helpers
# --------------------------------------------------------------------------


def line_reveal(index: int, count: int, progress: float, spread: float = 0.55) -> float:
    """Staggered 0..1 reveal for line `index` of `count`."""
    if count <= 1:
        return clamp(progress, 0.0, 1.0)
    start = (index / count) * spread
    span = 1.0 - spread
    if span <= 1e-6:
        return clamp(progress, 0.0, 1.0)
    return clamp((progress - start) / span, 0.0, 1.0)


def char_progress(index: int, length: int, reveal: float) -> float:
    """Left-to-right character wipe inside one line."""
    if length <= 1:
        return clamp(reveal, 0.0, 1.0)
    head = reveal * (length + 4) - 4.0
    return clamp((head - index) / 3.0, 0.0, 1.0)


# --------------------------------------------------------------------------
# Compositions
# --------------------------------------------------------------------------


def _centered(text: str, size: int, y: float, family: str = "sans",
              color=PALETTE.ink, tracking: float = 0.0) -> TextLine:
    font, actual = fit_text(text, CONFIG.width - 120, family, start_size=size, min_size=10)
    line = TextLine(
        text=text,
        position=Vec2(0.0, y),
        font=font,
        size=actual,
        color=color,
        alignment="center",
        family=family,
        tracking=tracking,
    )
    line.position = Vec2((CONFIG.width - line.width) / 2.0, y)
    return line


def layout_intro(progress: float, disruption: float = 0.0) -> TextBlock:
    """Title card shown during the first 2.6 seconds."""
    progress = clamp(progress, 0.0, 1.0)
    block = TextBlock(phase="intro", progress=progress)

    name_line = _centered(PROFILE.name, 92, 168, "sans", PALETTE.ink)
    headline = _centered(PROFILE.headline, 22, 276, "sans", PALETTE.ink_dim, tracking=2.4)
    tagline = _centered(PROFILE.tagline, 14, 316, "mono", PALETTE.ink_faint, tracking=1.6)
    handle = _centered("@" + PROFILE.handle, 15, 430, "mono", PALETTE.ink_dim, tracking=1.2)

    for index, line in enumerate((name_line, headline, tagline, handle)):
        reveal = line_reveal(index, 4, progress)
        line.reveal = reveal
        line.weight = 0.35 + 0.65 * reveal
        line.offset = Vec2(0.0, lerp(22.0, 0.0, ease_out_cubic(reveal)))
        block.lines.append(line)

    # A thin rule that draws itself under the name.
    rule = TextLine(
        text="",
        position=Vec2(0.0, 250.0),
        font=load_font(12, "mono"),
        size=12,
        color=PALETTE.ink_faint,
        alignment="center",
        reveal=progress,
    )
    block.lines.append(rule)

    if disruption > 0.0:
        for line in block.lines:
            line.offset = line.offset + Vec2(
                smooth_noise(hash_text(line.text) * 0.01, 1.0, disruption * 4.0) * 16.0 * disruption,
                smooth_noise(hash_text(line.text) * 0.01, 1.0, disruption * 9.0) * 9.0 * disruption,
            )
    return block


def hash_text(text: str) -> int:
    """Stable per-string hash (Python's hash() is salted per process)."""
    value = 2166136261
    for char in text:
        value = (value ^ ord(char)) * 16777619
        value &= 0xFFFFFFFF
    return value


def layout_finale(progress: float, disruption: float = 0.0) -> TextBlock:
    """Full GitHub profile composition shown from 9.4s to the loop point."""
    progress = clamp(progress, 0.0, 1.0)
    block = TextBlock(phase="finale", progress=progress)

    margin = 68
    max_width = CONFIG.width - margin * 2

    def add(text: str, y: float, size: int, family: str, color, align: Alignment = "left",
            tracking: float = 0.0, uppercase: bool = False) -> None:
        content = text.upper() if uppercase else text
        font, actual = fit_text(content, max_width, family, start_size=size, min_size=9)
        line = TextLine(
            text=content,
            position=Vec2(margin if align == "left" else 0.0, y),
            font=font,
            size=actual,
            color=color,
            alignment=align,
            family=family,
            tracking=tracking,
        )
        if align == "center":
            line.position = Vec2((CONFIG.width - line.width) / 2.0, y)
        elif align == "right":
            line.position = Vec2(CONFIG.width - margin - line.width, y)
        block.lines.append(line)

    add(PROFILE.name, 54, 56, "sans", PALETTE.ink)
    add(PROFILE.headline, 118, 16, "sans", PALETTE.ink_dim, tracking=1.6, uppercase=True)
    add(PROFILE.description, 146, 13, "sans", PALETTE.ink_faint)

    add(PROFILE.research_heading, 196, 15, "mono", PALETTE.accent_blue, tracking=1.4, uppercase=True)
    add(" · ".join(PROFILE.research), 220, 14, "sans", PALETTE.ink)

    add(PROFILE.projects_heading, 262, 15, "mono", PALETTE.accent_blue, tracking=1.4, uppercase=True)
    add(" · ".join(label for label, _ in PROFILE.projects), 286, 14, "sans", PALETTE.ink)

    add(PROFILE.engineering_heading, 328, 15, "mono", PALETTE.accent_blue, tracking=1.4, uppercase=True)
    add(" · ".join(value for _, value in PROFILE.engineering), 352, 14, "sans", PALETTE.ink)

    add(PROFILE.connect_heading, 394, 15, "mono", PALETTE.accent_blue, tracking=1.4, uppercase=True)
    add(PROFILE.footer, 418, 16, "mono", PALETTE.ink, tracking=2.0, uppercase=True)

    add(PROFILE.footer_sub, 476, 11, "mono", PALETTE.ink_faint, align="center", tracking=3.0)

    # Staggered reveal, top to bottom.
    count = len(block.lines)
    for index, line in enumerate(block.lines):
        reveal = line_reveal(index, count, progress, spread=0.62)
        line.reveal = reveal
        line.weight = 0.25 + 0.75 * reveal
        line.offset = Vec2(0.0, lerp(16.0, 0.0, ease_out_cubic(reveal)))

    if disruption > 0.0:
        for line in block.lines:
            seed = hash_text(line.text) * 0.001
            line.offset = line.offset + Vec2(
                smooth_noise(seed, 1.0, disruption * 5.0) * 20.0 * disruption,
                smooth_noise(seed, 1.0, disruption * 11.0) * 11.0 * disruption,
            )
    return block


def layout_for_time(t: float, disruption: float = 0.0) -> TextBlock:
    """Pick the composition that owns absolute time t."""
    t = loop_time(t)
    state = evaluate_timeline(t)
    if state.typography_phase == "intro":
        return layout_intro(state.typography_progress, disruption)
    if state.typography_phase == "finale":
        return layout_finale(state.typography_progress, disruption)
    return TextBlock(phase="none", progress=0.0)


# --------------------------------------------------------------------------
# Safe zones
# --------------------------------------------------------------------------


def text_safe_zones(t: float) -> List[Tuple[int, int, int, int]]:
    """
    Rectangles occupied by profile text at absolute time t.

    The composition renderer (src/render.py / src/effects.py) uses
    these to keep the tree, roots and fighters out of the copy.
    Typography does not clip anything itself.
    """
    t = loop_time(t)
    state = evaluate_timeline(t)
    block = layout_for_time(t, disruption=0.0)
    if not block.lines or block.phase == "none":
        return []

    x0, y0, x1, y1 = block.bounds
    pad = 18
    rect = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(CONFIG.width, x1 + pad),
        min(CONFIG.height, y1 + pad),
    )

    if block.phase == "intro":
        strength = 1.0 - smoothstep(TYPOGRAPHY_INTRO_END - 0.5, TYPOGRAPHY_INTRO_END, t)
    else:
        strength = smoothstep(TYPOGRAPHY_FINALE_START, TYPOGRAPHY_FINALE_START + 0.6, t)

    zones = [SafeZone(rect=rect, phase=block.phase, strength=clamp(strength, 0.0, 1.0))]

    # Footer strip is always protected once the finale has landed.
    if block.phase == "finale" and strength > 0.5:
        zones.append(
            SafeZone(
                rect=(0, CONFIG.height - 52, CONFIG.width, CONFIG.height),
                phase="footer",
                strength=strength,
            )
        )
    return [z.as_tuple() for z in zones]


def safe_zone_rects(t: float) -> List[SafeZone]:
    """Same data as text_safe_zones() but as SafeZone objects."""
    return [SafeZone(rect=r, phase="text", strength=1.0) for r in text_safe_zones(t)]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _draw_line(
    draw: ImageDraw.ImageDraw,
    line: TextLine,
    disruption: float,
) -> None:
    if line.text == "":
        # The rule under the name: a line that wipes on with the reveal.
        if line.reveal <= 0.0:
            return
        width = int(360 * line.reveal)
        x = (CONFIG.width - width) / 2.0
        y = line.position.y + line.offset.y
        draw.line([(x, y), (x + width, y)], fill=line.color, width=1)
        return

    base_alpha = clamp(line.alpha * line.weight, 0.0, 1.0)
    if base_alpha <= 0.01:
        return

    colour = (
        int(line.color[0]),
        int(line.color[1]),
        int(line.color[2]),
        int(clamp(base_alpha * 255, 0, 255)),
    )

    cursor = line.position.x + line.offset.x
    y = line.position.y + line.offset.y
    length = len(line.text)

    for index, char in enumerate(line.text):
        if char == " ":
            cursor += text_width(" ", line.font) + line.tracking
            continue

        char_alpha = char_progress(index, length, line.reveal)
        if char_alpha <= 0.01:
            cursor += text_width(char, line.font) + line.tracking
            continue

        dx = 0.0
        dy = 0.0
        if disruption > 0.0:
            seed = (hash_text(line.text) + index * 7919) * 0.001
            dx = smooth_noise(seed, 1.0, disruption * 6.0) * 26.0 * disruption
            dy = smooth_noise(seed, 1.0, disruption * 13.0) * 14.0 * disruption

        fill = colour if char_alpha >= 0.99 else (
            colour[0], colour[1], colour[2],
            int(colour[3] * char_alpha),
        )
        draw.text((cursor + dx, y + dy), char, font=line.font, fill=fill)
        cursor += text_width(char, line.font) + line.tracking


def _speckle(
    draw: ImageDraw.ImageDraw,
    block: TextBlock,
    disruption: float,
) -> None:
    """Ink flecks shed by the text while it fragments."""
    if disruption <= 0.02:
        return
    x0, y0, x1, y1 = block.bounds
    if x1 <= x0 or y1 <= y0:
        return
    count = int(60 * disruption)
    for index in range(count):
        seed = (hash_text(block.phase) + index * 104729) * 0.001
        px = lerp(x0, x1, (math.sin(seed * 12.9898) * 0.5 + 0.5))
        py = lerp(y0, y1, (math.sin(seed * 78.233) * 0.5 + 0.5))
        px += smooth_noise(seed, 1.0, disruption * 3.0) * 40.0 * disruption
        py += smooth_noise(seed, 1.0, disruption * 8.0) * 24.0 * disruption
        radius = 1.0 + 2.4 * disruption
        alpha = int(clamp(180 * disruption, 0, 255))
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill=(PALETTE.ink[0], PALETTE.ink[1], PALETTE.ink[2], alpha),
        )


def render_typography(
    t: float,
    size: Tuple[int, int] = None,
    disruption: float = None,
) -> Image.Image:
    """
    Render the profile typography for absolute time t into an RGBA layer.

    `disruption` defaults to the choreography curve, which spikes on
    every impact and scatters the letters into ink fragments.
    """
    size = size or CONFIG.size()
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    state = evaluate_timeline(t)
    if state.typography_phase == "none":
        return layer

    if disruption is None:
        disruption = state.typography_disruption * (0.35 if state.typography_phase == "intro" else 1.0)
    disruption = clamp(disruption, 0.0, 1.0)

    block = layout_for_time(t, disruption)
    if not block.lines:
        return layer

    # Scale from the 960x540 design canvas to the requested size.
    scale = size[0] / CONFIG.width
    if abs(scale - 1.0) > 1e-6:
        design = Image.new("RGBA", CONFIG.size(), (0, 0, 0, 0))
        draw = ImageDraw.Draw(design)
        for line in block.lines:
            _draw_line(draw, line, disruption)
        _speckle(draw, block, disruption)
        return design.resize(size, Image.LANCZOS)

    draw = ImageDraw.Draw(layer)
    for line in block.lines:
        _draw_line(draw, line, disruption)
    _speckle(draw, block, disruption)
    return layer


def render_typography_at_time(
    frame: Image.Image,
    time: float,
    disruption: float = None,
) -> Image.Image:
    """Composite typography onto `frame`. Screen space — no camera pass."""
    layer = render_typography(time, size=frame.size, disruption=disruption)
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    frame.alpha_composite(layer)
    return frame


# --------------------------------------------------------------------------
# Standalone preview
# --------------------------------------------------------------------------

PREVIEW_TIMES: Tuple[float, ...] = (0.6, 1.7, 2.4, 5.48, 10.1, 11.7)


def render_preview(
    times: Sequence[float] = PREVIEW_TIMES,
    panel_size: Tuple[int, int] = (480, 270),
    out_path: Optional[Path] = None,
) -> Image.Image:
    """Six frames: intro reveal, fragmentation and the finale reveal."""
    columns, rows = 3, 2
    sheet = Image.new(
        "RGB",
        (panel_size[0] * columns, panel_size[1] * rows),
        CONFIG.background,
    )
    for index, moment in enumerate(times):
        reference = Image.new("RGBA", CONFIG.size(), CONFIG.background + (255,))
        render_typography_at_time(reference, moment)

        draw = ImageDraw.Draw(reference)
        state = evaluate_timeline(moment)
        draw.text((8, 6), f"t={moment:.2f}s {state.typography_phase.upper()} "
                          f"dis={state.typography_disruption:.2f}",
                  fill=PALETTE.ink_faint)
        for rect in text_safe_zones(moment):
            draw.rectangle(rect, outline=PALETTE.accent_red)

        panel = reference.convert("RGB").resize(panel_size, Image.LANCZOS)
        sheet.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))

    path = out_path or (OUTPUT_DIR / "typography_preview.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return sheet


# --------------------------------------------------------------------------
# Standalone tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    # Fonts load for every family and size.
    for family in ("sans", "serif", "mono"):
        font = load_font(32, family)
        assert font is not None
    assert load_font(32, "sans") is load_font(32, "sans"), "font cache must hit"
    assert load_font(48, "sans") is not load_font(32, "sans")

    # Adaptive fitting.
    big_font, big_size = fit_text("NAYAN.", 900, "sans", start_size=200, min_size=10)
    assert text_width("NAYAN.", big_font) <= 900
    narrow_font, narrow_size = fit_text("NAYAN.", 120, "sans", start_size=200, min_size=10)
    assert text_width("NAYAN.", narrow_font) <= 120
    assert narrow_size <= big_size

    # Profile content is wired in.
    assert PROFILE.handle == "nayan-mis"
    assert PROFILE.name
    assert len(PROFILE.research) >= 3
    assert len(PROFILE.projects) >= 2

    # Compositions.
    intro = layout_intro(1.0)
    assert intro.phase == "intro"
    assert intro.lines
    assert any(line.text == PROFILE.name for line in intro.lines)

    finale = layout_finale(1.0)
    assert finale.phase == "finale"
    joined = " ".join(line.text for line in finale.lines)
    for token in (PROFILE.name, PROFILE.footer, PROFILE.research_heading):
        assert token in joined, f"{token} missing from finale layout"

    # Reveal staggering: earlier lines finish first.
    early = layout_intro(0.3)
    late = layout_intro(0.9)
    assert early.lines[0].reveal <= late.lines[0].reveal
    assert late.lines[0].reveal >= late.lines[-1].reveal

    # Nothing is laid out during combat.
    assert layout_for_time(6.5).phase == "none"
    assert render_typography(6.5).getbbox() is None

    # Safe zones: present in both text acts, absent during combat.
    intro_zones = text_safe_zones(1.5)
    finale_zones = text_safe_zones(11.0)
    combat_zones = text_safe_zones(6.5)
    assert intro_zones, "intro must declare a safe zone"
    assert finale_zones, "finale must declare a safe zone"
    assert combat_zones == []
    for rect in intro_zones + finale_zones:
        x0, y0, x1, y1 = rect
        assert 0 <= x0 < x1 <= CONFIG.width
        assert 0 <= y0 < y1 <= CONFIG.height

    # SafeZone overlap helper.
    zone = SafeZone(rect=(10, 10, 100, 100), phase="text", strength=1.0)
    assert zone.overlaps((50, 50, 150, 150))
    assert not zone.overlaps((200, 200, 300, 300))
    assert zone.x0 == 10 and zone.y1 == 100

    # Deterministic rendering, in or out of order.
    a = render_typography(1.7)
    b = render_typography(1.7)
    assert a.tobytes() == b.tobytes()
    c = render_typography(11.0)
    assert c.getbbox() is not None

    # Fragmentation actually moves pixels.
    calm = render_typography(11.0, disruption=0.0)
    shattered = render_typography(11.0, disruption=1.0)
    assert calm.tobytes() != shattered.tobytes()

    # Impact disruption comes from the choreography curve.
    assert evaluate_timeline(5.48).typography_disruption > 0.4
    assert evaluate_timeline(1.0).typography_disruption == 0.0

    # Scaling to other sizes keeps the aspect ratio and still draws.
    scaled = render_typography(11.0, size=(480, 270))
    assert scaled.size == (480, 270)
    assert scaled.getbbox() is not None

    # Compositing onto a frame.
    frame = Image.new("RGBA", CONFIG.size(), PALETTE.background + (255,))
    before = frame.tobytes()
    after = render_typography_at_time(frame, 11.0).tobytes()
    assert after != before

    # Preview.
    sheet = render_preview()
    assert sheet.size == (480 * 3, 270 * 2)
    assert (OUTPUT_DIR / "typography_preview.png").exists()


if __name__ == "__main__":
    _self_test()
    print("All typography tests passed.")
    print("INKBOUND typography engine loaded.")
