"""
INKBOUND — Global Configuration
File: src/config.py

Single source of truth for canvas geometry, timing, palette,
scene boundaries, actor definitions, tree settings, export
targets and the GitHub profile content.

Every other module reads from CONFIG instead of hard-coding
numbers, so a layout change happens in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

Color = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Palette:
    """Monochrome-first palette with two accent inks."""

    background: Color = (8, 8, 8)
    ink: Color = (240, 240, 240)
    ink_dim: Color = (150, 150, 150)
    ink_faint: Color = (70, 70, 70)
    accent_blue: Color = (57, 186, 230)
    accent_red: Color = (255, 51, 102)
    accent_gold: Color = (255, 180, 84)
    vignette: Color = (0, 0, 0)


# --------------------------------------------------------------------------
# Scene boundaries — five acts across twelve seconds
# --------------------------------------------------------------------------

#   act 0  INTRO        0.0 – 3.0
#   act 1  INTRUSION    3.0 – 5.0
#   act 2  BATTLE       5.0 – 9.0
#   act 3  TREE GROWTH  9.0 – 11.0
#   act 4  FINALE      11.0 – 12.0

SCENE_INTRO_END = 3.0
SCENE_INTRUSION_END = 5.0
SCENE_BATTLE_END = 9.0
SCENE_GROWTH_END = 11.0
SCENE_FINALE_END = 12.0

# Backwards-compatible aliases used by earlier drafts of the modules.
SCENE_APPROACH_END = SCENE_INTRUSION_END
SCENE_CLASH_END = SCENE_BATTLE_END
SCENE_EXCHANGE_END = SCENE_BATTLE_END

SCENES: Tuple[Tuple[str, float, float], ...] = (
    ("intro", 0.0, SCENE_INTRO_END),
    ("intrusion", SCENE_INTRO_END, SCENE_INTRUSION_END),
    ("battle", SCENE_INTRUSION_END, SCENE_BATTLE_END),
    ("growth", SCENE_BATTLE_END, SCENE_GROWTH_END),
    ("finale", SCENE_GROWTH_END, SCENE_FINALE_END),
)

# Absolute impact timestamps in seconds (derived from the storyboard).
IMPACT_TIMES: Tuple[float, ...] = (5.48, 6.5, 8.25)

# Typography cue windows.
TYPOGRAPHY_INTRO_END = 2.6
TYPOGRAPHY_FINALE_START = 9.4

# Tree growth window.
TREE_GROWTH_START = 9.0
TREE_GROWTH_END = 11.0

# Ground line in reference pixels.
GROUND_Y = 392.0

# Clash point — where the blades meet and the tree erupts.
CLASH_POINT = (480.0, 206.0)
TREE_ORIGIN = (480.0, GROUND_Y)


# --------------------------------------------------------------------------
# Profile content — edit here to re-brand the whole animation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """The GitHub identity rendered by src/typography.py."""

    handle: str = "nayan-mis"
    name: str = "NAYAN."
    headline: str = "COMPUTER SCIENCE / ARTIFICIAL INTELLIGENCE / SYSTEMS"
    tagline: str = "AN EXPERIMENT IN CODE, MOTION, AND IMAGINATION."
    description: str = (
        "Exploring intelligent systems, machine learning, software "
        "engineering, and the boundaries between research and building."
    )

    research_heading: str = "01 / RESEARCH"
    research: List[str] = field(
        default_factory=lambda: [
            "Artificial Intelligence",
            "Machine Learning",
            "Computer Systems",
            "Human-Computer Interaction",
        ]
    )

    projects_heading: str = "02 / PROJECTS"
    projects: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("Velune AI", "AI-assisted skincare and ingredient education."),
            ("Repositories", "Software experiments, prototypes and engineering."),
            ("INKBOUND", "Procedural animation and cinematic rendering."),
        ]
    )

    engineering_heading: str = "03 / ENGINEERING"
    engineering: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("LANGUAGES", "Python / TypeScript"),
            ("APPLICATIONS", "Web / APIs / Interactive Systems"),
            ("AI", "Machine Learning / Agents / Evaluation"),
        ]
    )

    principles_heading: str = "04 / PRINCIPLES"
    principles: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("Understand the fundamentals.", "Build from first principles."),
            ("Experiment deliberately.", "Turn ideas into testable systems."),
            ("Design with intent.", "Care about behaviour and reliability."),
        ]
    )

    connect_heading: str = "05 / CONNECT"
    footer: str = "GITHUB.COM/NAYAN-MIS"
    footer_sub: str = "INKBOUND / 2026"


# --------------------------------------------------------------------------
# Actors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorConfig:
    """Static description of one fighter."""

    key: str
    label: str
    color: Color
    accent: Color
    home: Tuple[float, float]
    facing: int  # +1 faces right, -1 faces left
    scale: float = 1.0


WANDERER = ActorConfig(
    key="wanderer",
    label="Wanderer",
    color=(240, 240, 240),
    accent=(57, 186, 230),
    home=(300.0, GROUND_Y - 86.0),
    facing=1,
    scale=1.0,
)

TWISTED_MAN = ActorConfig(
    key="twisted_man",
    label="Twisted Man",
    color=(240, 240, 240),
    accent=(255, 51, 102),
    home=(660.0, GROUND_Y - 86.0),
    facing=-1,
    scale=1.0,
)

ACTORS: Dict[str, ActorConfig] = {a.key: a for a in (WANDERER, TWISTED_MAN)}


# --------------------------------------------------------------------------
# Tree
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeConfig:
    """Settings consumed by src/tree.py."""

    seed: int = 20260101
    max_depth: int = 7
    trunk_length: float = 150.0
    origin: Tuple[float, float] = TREE_ORIGIN
    growth_start: float = TREE_GROWTH_START
    growth_end: float = TREE_GROWTH_END
    leaf_count: int = 90
    leaf_size: Tuple[float, float] = (2.5, 6.0)
    root_spokes: int = 5


TREE = TreeConfig()


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportConfig:
    """Targets used by src/export.py."""

    width: int = 960
    height: int = 540
    fps: int = 20
    duration: float = 12.0
    quality: int = 75
    method: int = 6
    target_mb: float = 10.0
    webp_name: str = "inkbound.webp"
    gif_name: str = "inkbound.gif"
    report_name: str = "export_report.json"


# --------------------------------------------------------------------------
# Root configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    # --- geometry ---------------------------------------------------------
    width: int = 960
    height: int = 540
    design_size: Tuple[int, int] = (960, 540)

    # --- timing -----------------------------------------------------------
    fps: int = 20
    duration: float = 12.0

    # --- rendering --------------------------------------------------------
    supersample: int = 2
    background: Color = Palette.background
    seed: int = 20260101

    # --- content ----------------------------------------------------------
    profile: Profile = field(default_factory=Profile)
    export: ExportConfig = field(default_factory=ExportConfig)
    palette: Palette = field(default_factory=Palette)
    tree: TreeConfig = field(default_factory=TreeConfig)
    actors: Dict[str, ActorConfig] = field(default_factory=lambda: dict(ACTORS))

    # ------------------------------------------------------------------
    @property
    def total_frames(self) -> int:
        return int(round(self.fps * self.duration))

    def frame_time(self, frame: int) -> float:
        """Absolute time in seconds for a frame index."""
        return frame / float(self.fps)

    def frame_index(self, time: float) -> int:
        """Frame index for an absolute time in seconds."""
        return int(round(time * self.fps))

    def size(self, scale: float = 1.0) -> Tuple[int, int]:
        return (int(round(self.width * scale)), int(round(self.height * scale)))

    def as_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "total_frames": self.total_frames,
            "supersample": self.supersample,
            "seed": self.seed,
            "profile": self.profile.handle,
        }


CONFIG = Config()

# Frequently used shortcuts so callers do not have to import Config.
WIDTH = CONFIG.width
HEIGHT = CONFIG.height
FPS = CONFIG.fps
DURATION = CONFIG.duration
TOTAL_FRAMES = CONFIG.total_frames
SEED = CONFIG.seed
PALETTE = CONFIG.palette
PROFILE = CONFIG.profile
TREE_CONFIG = CONFIG.tree


def loop_time(t: float) -> float:
    """Wrap absolute time into [0, duration)."""
    import math

    if DURATION <= 0:
        return 0.0
    t = math.fmod(t, DURATION)
    if t < 0:
        t += DURATION
    return t


def scene_at(t: float) -> Tuple[str, float, float]:
    """Return (name, start, end) for absolute time t (looped)."""
    t = loop_time(t)
    for name, start, end in SCENES:
        if start <= t < end:
            return (name, start, end)
    return SCENES[-1]


def validate() -> List[str]:
    """Return a list of configuration problems (empty when healthy)."""
    problems: List[str] = []
    if CONFIG.width <= 0 or CONFIG.height <= 0:
        problems.append("canvas dimensions must be positive")
    if CONFIG.fps <= 0:
        problems.append("fps must be positive")
    if CONFIG.duration <= 0:
        problems.append("duration must be positive")
    if CONFIG.total_frames < 1:
        problems.append("total_frames must be at least 1")
    if not PROFILE.name.strip():
        problems.append("profile.name must not be empty")
    previous = -1e-9
    for _name, start, end in SCENES:
        if end <= start:
            problems.append(f"scene {_name} has an empty or reversed range")
        if start < previous:
            problems.append(f"scene {_name} overlaps the previous scene")
        previous = end
    if abs(SCENES[-1][2] - DURATION) > 1e-9:
        problems.append("scene timeline must end at CONFIG.duration")
    return problems


if __name__ == "__main__":
    issues = validate()
    if issues:
        raise SystemExit("CONFIG errors: " + "; ".join(issues))
    print("All configuration tests passed.")
    print("INKBOUND config loaded.")
    for key, value in CONFIG.as_dict().items():
        print(f"  {key}: {value}")
    for name, start, end in SCENES:
        print(f"  scene {name}: {start:.2f}s -> {end:.2f}s")
