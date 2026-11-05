"""
INKBOUND — package marker.

Makes `src` importable so `python -m src.render`, `python -m src.export`
and `from src.effects import compose_scene` all work from the project root.
"""

from __future__ import annotations

__all__ = [
    "config",
    "characters",
    "choreography",
    "tree",
    "particles",
    "typography",
    "effects",
    "render",
    "export",
]
