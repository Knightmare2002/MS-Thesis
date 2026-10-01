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

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES  # noqa: E402

DAMAGE_CLASSES: tuple[str, ...] = tuple(UNIFIED_DAMAGE_CLASSES)
NUM_CLASSES: int = len(DAMAGE_CLASSES)

# Order verified in the repository on 2026-09-30. If the taxonomy changes, this
# guard fails loudly instead of silently permuting the 3D channels.
_EXPECTED_ORDER = ("crack", "spalling", "corrosion", "moisture", "delamination", "surface")
if DAMAGE_CLASSES != _EXPECTED_ORDER:
    raise RuntimeError(
        f"UNIFIED_DAMAGE_CLASSES changed: {DAMAGE_CLASSES} != {_EXPECTED_ORDER}. "
        "Update damage3d (colors, docs, tests) before running the 3D pipeline."
    )

# Visual convention (RGB, 0-255). Distinct from the state colors below.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "crack": (228, 26, 28),          # red
    "spalling": (255, 127, 0),       # orange
    "corrosion": (166, 86, 40),      # brown
    "moisture": (0, 190, 255),       # cyan
    "delamination": (152, 78, 163),  # purple
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
