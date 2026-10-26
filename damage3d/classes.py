"""Damage class order, taken from the thesis single source of truth.

The channel order is imported from ``src/data/class_mapping.py``
(``UNIFIED_DAMAGE_CLASSES``) so that the 3D pipeline cannot drift from the
P1ML training/evaluation code. Channels are independent (multilabel): no
softmax and no mandatory argmax anywhere in this package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.class_mapping import CLOSED_DAMAGE_CLASSES, UNIFIED_DAMAGE_CLASSES  # noqa: E402

# MODEL_CLASSES: channels written by the network / scripts 23-28 (maps, Platt, thresholds).
# DAMAGE_CLASSES: channels mapped in 3D = MODEL_CLASSES minus the closed classes
# (delamination since 2026-10-26). Providers select them BY NAME from the maps.
MODEL_CLASSES: tuple[str, ...] = tuple(UNIFIED_DAMAGE_CLASSES)
CLOSED_CLASSES: tuple[str, ...] = tuple(CLOSED_DAMAGE_CLASSES)
DAMAGE_CLASSES: tuple[str, ...] = tuple(c for c in MODEL_CLASSES if c not in CLOSED_CLASSES)
NUM_CLASSES: int = len(DAMAGE_CLASSES)
MODEL_CHANNELS: tuple[int, ...] = tuple(MODEL_CLASSES.index(c) for c in DAMAGE_CLASSES)

# Orders verified in the repository (2026-09-30, closure 2026-10-26). If the
# taxonomy changes, this guard fails loudly instead of silently permuting channels.
_EXPECTED_MODEL_ORDER = ("crack", "spalling", "corrosion", "moisture", "delamination", "surface")
_EXPECTED_ORDER = ("crack", "spalling", "corrosion", "moisture", "surface")
if MODEL_CLASSES != _EXPECTED_MODEL_ORDER or DAMAGE_CLASSES != _EXPECTED_ORDER:
    raise RuntimeError(
        f"Class order changed: model {MODEL_CLASSES} / 3D {DAMAGE_CLASSES}, expected "
        f"{_EXPECTED_MODEL_ORDER} / {_EXPECTED_ORDER}. Update damage3d (colors, docs, tests) first."
    )

# Visual convention (RGB, 0-255). Distinct from the state colors below.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "crack": (228, 26, 28),          # red
    "spalling": (255, 127, 0),       # orange
    "corrosion": (166, 86, 40),      # brown
    "moisture": (0, 190, 255),       # cyan
    "delamination": (152, 78, 163),  # purple (closed: only in 2D overlays of model channels)
    "surface": (77, 175, 74),        # green
}

# Point observation states (stored as uint8 in every output).
STATE_NOT_OBSERVED = 0         # no valid view at all: never colored as healthy
STATE_INSUFFICIENT_VIEWS = 1   # seen, but not enough valid views for every class
STATE_OBSERVED_NO_DAMAGE = 2   # every class has >= min_views valid views, no label
STATE_DAMAGE = 3               # at least one class label is positive

STATE_NAMES = {
    STATE_NOT_OBSERVED: "not_observed",
    STATE_INSUFFICIENT_VIEWS: "insufficient_views",
    STATE_OBSERVED_NO_DAMAGE: "observed_no_damage",
    STATE_DAMAGE: "damage",
}

STATE_COLORS: dict[int, tuple[int, int, int]] = {
    STATE_NOT_OBSERVED: (40, 70, 160),         # dark blue
    STATE_INSUFFICIENT_VIEWS: (90, 90, 90),    # dark grey
    STATE_OBSERVED_NO_DAMAGE: (200, 200, 200), # light grey
}


def class_bit(name: str) -> int:
    """Return the bit value of a class in ``label_mask`` (crack=1, spalling=2, ...)."""
    return 1 << DAMAGE_CLASSES.index(name)
