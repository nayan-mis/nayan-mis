"""
INKBOUND — Production Animation Export Pipeline
File: src/export.py

Features:
    - Animated WebP and GIF export
    - Adaptive quality, FPS, and resolution
    - Disk-backed source-frame staging
    - Deterministic frame selection
    - Duration and loop verification
    - File-size validation
    - GIF fallback
    - JSON export reports
    - Standalone tests

Dependencies:
    Pillow
    src/config.py
    src/render.py

Examples:
    python -m src.export --test
    python -m src.export --draft
    python -m src.export --format webp
    python -m src.export --target-mb 8
    python -m src.export --format gif

Notes:
    Pillow's animation encoders may still load all selected
    frames into memory during encoding.  Disk staging reduces
    repeated rendering costs, but does not guarantee bounded
    encoder memory.

Design constraint: the source animation stays 12 seconds long.
Optimisation may reduce resolution, FPS or quality — never the
length of the choreography.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

try:
    from .config import CONFIG, OUTPUT_DIR
    from .render import render_frame
except ImportError:  # pragma: no cover - direct script execution
    from config import CONFIG, OUTPUT_DIR  # type: ignore
    from render import render_frame  # type: ignore

__all__ = [
    "FrameStage",
    "select_frame_indices",
    "verify_animation",
    "encode_frames",
    "adaptive_export",
    "export_report_path",
    "SOURCE_FPS",
]

SOURCE_FPS = CONFIG.fps
SOURCE_FRAMES = CONFIG.total_frames
TARGET_DURATION = CONFIG.duration

# The optimisation ladder, tried in order.
QUALITY_LADDER: Tuple[int, ...] = (75, 65, 55, 45, 35)
FPS_LADDER: Tuple[int, ...] = (20, 15, 12, 10)
RESOLUTION_LADDER: Tuple[float, ...] = (1.0, 0.85, 0.7)

DURATION_TOLERANCE = 0.25  # seconds


# --------------------------------------------------------------------------
# Frame selection
# --------------------------------------------------------------------------


def select_frame_indices(fps: int) -> List[int]:
    """
    Deterministically sample the 240-frame source for a target fps.

    The choreography is never shortened: selecting fewer frames
    keeps the full 12 seconds, just sampled less often.
    """
    fps = max(1, int(fps))
    count = max(2, int(round(TARGET_DURATION * fps)))
    last = SOURCE_FRAMES - 1
    indices: List[int] = []
    for k in range(count):
        index = int(round(k * last / (count - 1)))
        index = min(last, max(0, index))
        if not indices or indices[-1] != index:
            indices.append(index)
    if indices[-1] != last:
        indices.append(last)
    return indices


def resolution_size(scale: float) -> Tuple[int, int]:
    """Output pixel dimensions for a resolution scale factor."""
    return (
        max(16, int(round(CONFIG.width * scale))),
        max(16, int(round(CONFIG.height * scale))),
    )


# --------------------------------------------------------------------------
# Disk-backed staging
# --------------------------------------------------------------------------


class FrameStage:
    """
    Renders source frames once to a temporary directory.

    Every encoding attempt then re-reads from disk instead of
    re-running the renderer, so the adaptive ladder pays for the
    render a single time.
    """

    def __init__(
        self,
        indices: Sequence[int],
        draft: bool = False,
        scale: int = 1,
        root: Optional[Path] = None,
        keep: bool = False,
    ) -> None:
        self.indices = [int(i) for i in indices]
        self.draft = draft
        self.scale = scale
        self.keep = keep
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="inkbound_stage_"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.renders = 0
        self.loads = 0

    # -- paths ----------------------------------------------------------
    def path_for(self, index: int) -> Path:
        return self.root / f"src_{index:05d}.png"

    # -- access ---------------------------------------------------------
    def ensure(self, index: int) -> Path:
        path = self.path_for(index)
        if not path.exists():
            frame = render_frame(CONFIG.frame_time(index), draft=self.draft, scale=self.scale)
            frame.convert("RGB").save(path, format="PNG")
            self.renders += 1
        return path

    def load(self, index: int) -> Image.Image:
        path = self.ensure(index)
        self.loads += 1
        with Image.open(path) as handle:
            return handle.convert("RGB")

    def load_all(self, indices: Optional[Sequence[int]] = None,
                 size: Optional[Tuple[int, int]] = None) -> List[Image.Image]:
        frames: List[Image.Image] = []
        for index in (self.indices if indices is None else indices):
            frame = self.load(int(index))
            if size is not None and frame.size != size:
                frame = frame.resize(size, Image.LANCZOS)
            frames.append(frame)
        return frames

    def cleanup(self) -> None:
        if self.keep:
            return
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "FrameStage":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@dataclass
class Verification:
    path: Path
    format: str
    exists: bool
    bytes: int = 0
    frames: int = 0
    duration_s: float = 0.0
    frame_delays_ms: List[int] = field(default_factory=list)
    loop: Optional[int] = None
    size: Tuple[int, int] = (0, 0)
    duration_ok: bool = False
    error: str = ""

    @property
    def size_mb(self) -> float:
        return self.bytes / (1024 * 1024)

    @property
    def ok(self) -> bool:
        return self.exists and self.duration_ok and self.frames > 0 and not self.error


def verify_animation(path: Path, expected_duration: float = TARGET_DURATION) -> Verification:
    """Decode an animation and check frame count, timing and loop."""
    path = Path(path)
    if not path.exists():
        return Verification(path=path, format="?", exists=False, error="file not found")

    result = Verification(
        path=path,
        format=path.suffix.lstrip(".").upper(),
        exists=True,
        bytes=path.stat().st_size,
    )
    try:
        with Image.open(path) as probe:
            result.format = (probe.format or result.format).upper()
            result.size = probe.size
            result.frames = int(getattr(probe, "n_frames", 1) or 1)
            loop = probe.info.get("loop")
            result.loop = int(loop) if loop is not None else None

            delays: List[int] = []
            for index in range(result.frames):
                probe.seek(index)
                duration = probe.info.get("duration", 0)
                delays.append(int(duration or 0))
            result.frame_delays_ms = delays
            result.duration_s = sum(delays) / 1000.0
    except Exception as exc:  # pragma: no cover - corrupt file
        result.error = f"decode failed: {exc}"
        return result

    if result.frames <= 1:
        result.error = "not an animation"
    elif abs(result.duration_s - expected_duration) > DURATION_TOLERANCE:
        result.error = (
            f"duration {result.duration_s:.3f}s differs from {expected_duration}s "
            f"by more than {DURATION_TOLERANCE}s"
        )
    else:
        result.duration_ok = True
    return result


def export_report_path() -> Path:
    return OUTPUT_DIR / CONFIG.export.report_name


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def encode_frames(
    frames: Sequence[Image.Image],
    out_path: Path,
    fmt: str = "webp",
    quality: int = 75,
    method: int = 6,
    fps: int = SOURCE_FPS,
) -> Path:
    """Write a list of RGB frames as an animated WebP or GIF."""
    if not frames:
        raise ValueError("no frames to encode")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))

    if fmt == "gif":
        frames[0].save(
            out_path,
            format="GIF",
            save_all=True,
            append_images=list(frames[1:]),
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    else:
        frames[0].save(
            out_path,
            format="WEBP",
            save_all=True,
            append_images=list(frames[1:]),
            duration=duration_ms,
            loop=0,
            quality=int(quality),
            method=int(method),
        )
    return out_path


# --------------------------------------------------------------------------
# Adaptive export
# --------------------------------------------------------------------------


@dataclass
class ExportAttempt:
    stage: str
    fmt: str
    quality: int
    fps: int
    resolution_scale: float
    bytes: int
    duration_s: float
    frames: int
    duration_ok: bool
    error: str = ""


@dataclass
class ExportResult:
    path: Path
    fmt: str
    bytes: int
    resolution: Tuple[int, int]
    fps: int
    frames: int
    duration_s: float
    loop: Optional[int]
    quality: int
    method: int
    within_budget: bool
    attempts: List[ExportAttempt] = field(default_factory=list)
    report_path: Optional[Path] = None
    elapsed_s: float = 0.0

    @property
    def size_mb(self) -> float:
        return self.bytes / (1024 * 1024)

    def as_dict(self) -> Dict[str, object]:
        return {
            "file": str(self.path),
            "format": self.fmt,
            "bytes": self.bytes,
            "size_mb": round(self.size_mb, 3),
            "resolution": list(self.resolution),
            "fps": self.fps,
            "frames": self.frames,
            "duration_s": round(self.duration_s, 4),
            "loop": self.loop,
            "quality": self.quality,
            "method": self.method,
            "within_budget": self.within_budget,
            "target_mb": CONFIG.export.target_mb,
            "source": {
                "fps": SOURCE_FPS,
                "frames": SOURCE_FRAMES,
                "duration_s": TARGET_DURATION,
                "resolution": [CONFIG.width, CONFIG.height],
            },
            "elapsed_s": round(self.elapsed_s, 2),
            "attempts": [vars(a) for a in self.attempts],
        }


def adaptive_export(
    out_path: Optional[Path] = None,
    fmt: str = "webp",
    target_mb: float = None,
    quality: int = None,
    draft: bool = False,
    scale: int = 1,
    max_attempts: int = 14,
    progress: bool = True,
    report: bool = True,
    stage_root: Optional[Path] = None,
) -> ExportResult:
    """
    Encode the animation, walking the optimisation ladder until the
    file fits the size budget (or the ladder runs out).

    Stage 1  quality  75 -> 35
    Stage 2  fps      20 -> 15 -> 12 -> 10
    Stage 3  resolution 1.0 -> 0.85 -> 0.7
    Stage 4  combined reductions
    """
    started = _time.time()
    fmt = fmt.lower()
    if fmt not in ("webp", "gif"):
        raise ValueError(f"unsupported format {fmt!r}")
    target_mb = CONFIG.export.target_mb if target_mb is None else float(target_mb)
    budget = int(target_mb * 1024 * 1024)

    out_path = Path(out_path) if out_path else OUTPUT_DIR / (
        CONFIG.export.webp_name if fmt == "webp" else CONFIG.export.gif_name
    )

    if progress:
        print(f"INKBOUND export: format={fmt} target={target_mb:.1f} MiB draft={draft}")

    # Ladder of candidate configurations.
    candidates: List[Tuple[str, int, float, int]] = []
    base_quality = CONFIG.export.quality if quality is None else int(quality)

    if fmt == "gif":
        # GIF has no meaningful quality knob; go straight to fps/resolution.
        for res in RESOLUTION_LADDER:
            for fps in FPS_LADDER:
                candidates.append(("gif", 100, res, fps))
    else:
        for q in QUALITY_LADDER:
            q = min(base_quality, q) if quality is None else int(quality)
            candidates.append(("stage1-quality", q, 1.0, SOURCE_FPS))
        for fps in FPS_LADDER[1:]:
            candidates.append(("stage2-fps", min(base_quality, 55), 1.0, fps))
        for res in RESOLUTION_LADDER[1:]:
            candidates.append(("stage3-resolution", min(base_quality, 45), res, 15))
        for res in RESOLUTION_LADDER[1:]:
            for fps in FPS_LADDER[1:]:
                candidates.append(("stage4-combined", 35, res, fps))

    # De-duplicate while preserving order.
    seen = set()
    unique: List[Tuple[str, int, float, int]] = []
    for item in candidates:
        if item[1:] in seen:
            continue
        seen.add(item[1:])
        unique.append(item)
    candidates = unique[:max_attempts]

    # One staging pass over the union of frame indices we may need.
    needed = sorted({
        index
        for _stage, _q, _res, fps in candidates
        for index in select_frame_indices(fps)
    })
    stage = FrameStage(needed, draft=draft, scale=scale, root=stage_root)

    attempts: List[ExportAttempt] = []
    best: Optional[Tuple[Verification, Tuple[int, int], int, float, int]] = None

    try:
        for stage_name, q, res, fps in candidates:
            indices = select_frame_indices(fps)
            size = resolution_size(res)
            frames = stage.load_all(indices, size=size)

            candidate_path = out_path.with_name(
                f"_{out_path.stem}_{stage_name}_{q}_{int(res * 100)}_{fps}{out_path.suffix}"
            )
            try:
                encode_frames(frames, candidate_path, fmt=fmt,
                              quality=q, method=CONFIG.export.method, fps=fps)
            except Exception as exc:  # pragma: no cover - encoder refusal
                attempts.append(ExportAttempt(stage_name, fmt, q, fps, res, 0, 0.0,
                                              len(indices), False, str(exc)))
                continue

            verification = verify_animation(candidate_path)
            attempt = ExportAttempt(
                stage=stage_name,
                fmt=fmt,
                quality=q,
                fps=fps,
                resolution_scale=res,
                bytes=verification.bytes,
                duration_s=round(verification.duration_s, 4),
                frames=verification.frames,
                duration_ok=verification.duration_ok,
                error=verification.error,
            )
            attempts.append(attempt)

            if progress:
                flag = "ok " if verification.duration_ok else "BAD"
                print(f"  [{flag}] {stage_name:18s} q={q:3d} fps={fps:2d} "
                      f"res={res:.2f} -> {verification.size_mb:6.2f} MiB "
                      f"({verification.frames} frames, {verification.duration_s:.3f}s)")

            if not verification.ok:
                candidate_path.unlink(missing_ok=True)
                continue

            if best is None or verification.bytes < best[0].bytes:
                best = (verification, size, q, float(res), fps)
            candidate_path.unlink(missing_ok=True)
            if verification.bytes <= budget:
                break
    finally:
        stage.cleanup()

    if best is None:
        raise RuntimeError(
            "no valid animation could be encoded; all candidates failed verification"
        )

    verification, size, q, res, fps = best

    # Re-encode the winning configuration to the final path.
    indices = select_frame_indices(fps)
    final_stage = FrameStage(indices, draft=draft, scale=scale,
                             root=stage_root or (OUTPUT_DIR / ".stage"))
    try:
        frames = final_stage.load_all(indices, size=size)
        encode_frames(frames, out_path, fmt=fmt, quality=q,
                      method=CONFIG.export.method, fps=fps)
    finally:
        final_stage.cleanup()

    final_check = verify_animation(out_path)
    if not final_check.ok:
        raise RuntimeError(f"final export failed verification: {final_check.error}")

    result = ExportResult(
        path=out_path,
        fmt=fmt,
        bytes=final_check.bytes,
        resolution=final_check.size,
        fps=fps,
        frames=final_check.frames,
        duration_s=round(final_check.duration_s, 4),
        loop=final_check.loop,
        quality=q,
        method=CONFIG.export.method,
        within_budget=final_check.bytes <= budget,
        attempts=attempts,
        elapsed_s=round(_time.time() - started, 2),
    )

    if report:
        path = export_report_path()
        path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        result.report_path = path

    if progress:
        status = "within budget" if result.within_budget else "OVER budget (best effort)"
        print(f"  -> {out_path.name}: {result.size_mb:.2f} MiB, "
              f"{result.frames} frames, {result.duration_s:.3f}s, {status}")
        if result.report_path:
            print(f"  -> {result.report_path}")
    return result


# --------------------------------------------------------------------------
# Standalone tests
# --------------------------------------------------------------------------


def _self_test() -> None:
    import tempfile as _tempfile

    # Frame selection keeps the full duration at every fps.
    for fps in FPS_LADDER:
        indices = select_frame_indices(fps)
        assert indices == sorted(set(indices)), "indices must be sorted and unique"
        assert indices[0] == 0
        assert indices[-1] <= SOURCE_FRAMES - 1
        span = (indices[-1] - indices[0]) / SOURCE_FPS
        assert span >= TARGET_DURATION - (1.0 / fps) - 1e-6, (
            f"fps {fps} must still cover 12s, got {span:.2f}s"
        )
        assert len(indices) >= int(TARGET_DURATION * fps * 0.9)
    assert select_frame_indices(20) == list(range(SOURCE_FRAMES))

    # Resolution ladder.
    assert resolution_size(1.0) == (CONFIG.width, CONFIG.height)
    assert resolution_size(0.5) == (CONFIG.width // 2, CONFIG.height // 2)

    # Staging: renders once, then reads from disk.
    with _tempfile.TemporaryDirectory() as tmp:
        stage = FrameStage([0, 10, 20], draft=True, root=Path(tmp) / "stage")
        first = stage.load(10)
        renders_after_first = stage.renders
        second = stage.load(10)
        assert renders_after_first == 1, "second load must not re-render"
        assert stage.loads == 2
        assert first.size == CONFIG.size()
        assert first.tobytes() == second.tobytes()
        assert all(stage.path_for(i).exists() for i in (0, 10, 20))
        stage.cleanup()
        assert not (Path(tmp) / "stage").exists(), "cleanup must remove the staging directory"

    # Verification rejects a missing file and a static image.
    missing = verify_animation(Path("definitely_not_here.webp"))
    assert missing.exists is False and not missing.ok

    with _tempfile.TemporaryDirectory() as tmp:
        static = Path(tmp) / "static.webp"
        Image.new("RGB", (64, 64), (10, 10, 10)).save(static, format="WEBP")
        check = verify_animation(static)
        assert check.exists and check.frames == 1
        assert not check.ok, "a single-frame image is not an animation"

    # Round-trip: encode six staged frames, then verify timing.
    with _tempfile.TemporaryDirectory() as tmp:
        stage = FrameStage([0, 40, 80, 120, 160, 200], draft=True, root=Path(tmp))
        frames = stage.load_all()
        stage.cleanup()

        webp = Path(tmp) / "roundtrip.webp"
        encode_frames(frames, webp, fmt="webp", quality=60, fps=20)
        webp_check = verify_animation(webp, expected_duration=6 * (1000 / 20) / 1000)
        assert webp_check.ok, webp_check.error
        assert webp_check.format == "WEBP"
        assert webp_check.frames == 6
        assert webp_check.size == CONFIG.size()
        assert webp_check.loop in (0, None)

        gif = Path(tmp) / "roundtrip.gif"
        encode_frames(frames, gif, fmt="gif", fps=20)
        gif_check = verify_animation(gif, expected_duration=6 * (1000 / 20) / 1000)
        assert gif_check.ok, gif_check.error
        assert gif_check.format == "GIF"
        assert gif_check.frames == 6

        # Duration mismatch is caught.
        wrong = verify_animation(webp, expected_duration=12.0)
        assert not wrong.duration_ok
        assert "differs" in wrong.error

    # A short adaptive export produces a verified file and a report.
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "adaptive.webp"
        result = adaptive_export(
            out_path=out,
            fmt="webp",
            target_mb=50.0,
            quality=45,
            draft=True,
            progress=False,
            report=False,
            max_attempts=3,
            stage_root=Path(tmp) / "stage",
        )
        assert result.path.exists()
        assert result.bytes > 0
        assert result.frames >= 2
        assert abs(result.duration_s - TARGET_DURATION) <= DURATION_TOLERANCE
        assert result.attempts, "the ladder must record its attempts"
        assert isinstance(result.within_budget, bool)

        # The report is valid JSON with the required keys.
        report_file = export_report_path()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        payload = json.loads(report_file.read_text(encoding="utf-8"))
        for key in ("file", "format", "bytes", "resolution", "fps", "frames",
                    "duration_s", "loop", "quality", "within_budget", "attempts"):
            assert key in payload, f"report missing {key}"
        report_file.unlink(missing_ok=True)

    # Unsupported format is rejected.
    try:
        adaptive_export(fmt="mp4", progress=False, report=False)
    except ValueError:
        pass
    else:
        raise AssertionError("mp4 must be rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INKBOUND production export")
    parser.add_argument("--test", action="store_true", help="run export tests")
    parser.add_argument("--draft", action="store_true", help="fast low-cost render")
    parser.add_argument("--format", choices=("webp", "gif"), default="webp")
    parser.add_argument("--target-mb", type=float, default=None)
    parser.add_argument("--quality", type=int, default=None)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=14)
    args = parser.parse_args()

    if args.test:
        _self_test()
        print("All export tests passed.")
        print("INKBOUND export pipeline loaded.")
        raise SystemExit(0)

    adaptive_export(
        fmt=args.format,
        target_mb=args.target_mb,
        quality=args.quality,
        draft=args.draft,
        scale=args.scale,
        max_attempts=args.max_attempts,
        progress=True,
        report=True,
    )
