"""Class taxonomies and the cross-dataset mapping used throughout the thesis.

Why this file exists
--------------------
CrackSeg9k is a *binary* crack segmentation dataset, dacl10k is a *multi-label*
bridge damage + component segmentation dataset, CODEBRIM is a multi-label
*classification* dataset. To train a shared-encoder / multi-head network we need
one explicit, versioned mapping instead of ad-hoc strings scattered in scripts.

Design decisions
----------------
1. `Crack` and `ACrack` (alligator crack) are both mapped to the unified
   `crack` class: geometrically distinct but semantically cracks, and this is
   what makes CrackSeg9k comparable with dacl10k.
2. dacl10k objects (Bearing, EJoint, ...) are kept in a separate group: they are
   *components*, not damages, and belong to a different decoder head.
3. Nothing is silently dropped: unmapped labels raise or are counted, never
   ignored without a trace.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# dacl10k: 19 classes, official order (index = channel index in the mask stack)
# --------------------------------------------------------------------------- #
DACL10K_CLASSES: list[str] = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket",
    "Hollowareas", "Cavity", "Spalling", "Graffiti", "Weathering",
    "Restformwork", "ExposedRebars", "Bearing", "EJoint", "Drainage",
    "PEquipment", "JTape", "WConccor",
]

DACL10K_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(DACL10K_CLASSES)}

# Damage vs. object (component) split, as defined in the dacl10k paper.
DACL10K_DAMAGE: list[str] = DACL10K_CLASSES[:13]
DACL10K_OBJECTS: list[str] = DACL10K_CLASSES[13:]

# --------------------------------------------------------------------------- #
# Unified taxonomy (thesis-level)
# --------------------------------------------------------------------------- #
# Branch A - binary crack segmentation (CrackSeg9k + dacl10k crack channels).
CRACK_LIKE_DACL10K: list[str] = ["Crack", "ACrack"]

# Branch B - damage classes kept for the multi-label head. Rare / non-structural
# classes are grouped to limit extreme class imbalance in a 4-month project.
UNIFIED_DAMAGE_CLASSES: list[str] = [
    "crack",          # Crack, ACrack
    "spalling",       # Spalling, Rockpocket, Cavity
    "corrosion",      # Rust, ExposedRebars, WConccor
    "moisture",       # Wetspot, Efflorescence
    "delamination",   # Hollowareas
    "surface",        # Weathering, Graffiti, Restformwork
]

DACL10K_TO_UNIFIED: dict[str, str] = {
    "Crack": "crack",
    "ACrack": "crack",
    "Spalling": "spalling",
    "Rockpocket": "spalling",
    "Cavity": "spalling",
    "Rust": "corrosion",
    "ExposedRebars": "corrosion",
    "WConccor": "corrosion",
    "Wetspot": "moisture",
    "Efflorescence": "moisture",
    "Hollowareas": "delamination",
    "Weathering": "surface",
    "Graffiti": "surface",
    "Restformwork": "surface",
    # Components are intentionally NOT part of the damage taxonomy.
    "Bearing": None,
    "EJoint": None,
    "Drainage": None,
    "PEquipment": None,
    "JTape": None,
}

# CrackSeg9k is single-class: its positive pixels map directly to `crack`.
CRACKSEG9K_TO_UNIFIED: dict[str, str] = {"crack": "crack"}

# Placeholder for week 4+ (CODEBRIM is classification, not segmentation).
CODEBRIM_TO_UNIFIED: dict[str, str] = {
    "Crack": "crack",
    "Spallation": "spalling",
    "Efflorescence": "moisture",
    "ExposedBars": "corrosion",
    "CorrosionStain": "corrosion",
}


def unified_damage_index() -> dict[str, int]:
    """Return {unified class name -> channel index} for the multi-label head."""
    return {c: i for i, c in enumerate(UNIFIED_DAMAGE_CLASSES)}


def dacl10k_channels_for(labels: list[str]) -> list[int]:
    """Map dacl10k label names to their channel indices, failing loudly on typos."""
    unknown = [lab for lab in labels if lab not in DACL10K_CLASS_TO_IDX]
    if unknown:
        raise KeyError(f"Unknown dacl10k labels: {unknown}")
    return [DACL10K_CLASS_TO_IDX[lab] for lab in labels]
