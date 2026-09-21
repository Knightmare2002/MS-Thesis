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
   `crack` class: geometrically distinct but semantically cracks, and this is what makes CrackSeg9k comparable with dacl10k.
2. dacl10k objects (Bearing, EJoint, ...) are kept in a separate group: they are *components*, not damages, and belong to a different decoder head.
3. Nothing is silently dropped: unmapped labels raise or are counted, never ignored without a trace.
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

# Placeholder for possible extensions (CODEBRIM is classification, not segmentation).
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


# --------------------------------------------------------------------------- #
# P1ML: DACL10K 19 -> 6 unified damage channels (single source of truth)
# --------------------------------------------------------------------------- #
# The multilabel pipeline is damage-only: bridge components are annotated in DACL10K but are not damages, so they are excluded from the target tensor instead of being folded into a residual channel (a "component" channel would make the macro Dice reward object detection, not damage detection).
DACL10K_EXCLUDED_FROM_DAMAGE: list[str] = [
    "Bearing", "EJoint", "Drainage", "PEquipment", "JTape",
]


def unified_damage_label_groups() -> dict[str, list[str]]:
    """Return {unified damage class -> DACL10K labels}, built from DACL10K_TO_UNIFIED.

    This is the only place where the 19 -> 6 grouping is materialised: datasets,
    losses, metrics and scripts must derive their channel order from here so a
    change to the taxonomy cannot desynchronise them.
    """
    groups: dict[str, list[str]] = {name: [] for name in UNIFIED_DAMAGE_CLASSES}

    for label in DACL10K_CLASSES:
        if label not in DACL10K_TO_UNIFIED:
            raise KeyError(
                f"DACL10K label '{label}' has no entry in DACL10K_TO_UNIFIED: "
                "the taxonomy is incomplete."
            )

        unified = DACL10K_TO_UNIFIED[label]
        if unified is None:
            continue

        if unified not in groups:
            raise KeyError(
                f"DACL10K_TO_UNIFIED maps '{label}' to unknown unified class "
                f"'{unified}'."
            )

        groups[unified].append(label)

    empty = [name for name, labels in groups.items() if not labels]
    if empty:
        raise RuntimeError(f"Unified damage classes without DACL10K labels: {empty}")

    return groups


def unified_damage_label_to_channel() -> dict[str, int]:
    """Return {DACL10K label -> unified damage channel index}, excluded labels absent."""
    groups = unified_damage_label_groups()
    return {
        label: channel
        for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES)
        for label in groups[name]
    }


def unified_damage_channel_groups() -> list[list[int]]:
    """Return, per unified channel, the DACL10K channel indices merged into it."""
    groups = unified_damage_label_groups()
    return [dacl10k_channels_for(groups[name]) for name in UNIFIED_DAMAGE_CLASSES]


def assert_unified_damage_taxonomy() -> None:
    """Fail loudly if the 19 -> 6 mapping is inconsistent (used by the smoke test)."""
    groups = unified_damage_label_groups()
    mapped = [label for labels in groups.values() for label in labels]

    duplicated = {label for label in mapped if mapped.count(label) > 1}
    if duplicated:
        raise RuntimeError(f"Labels assigned to more than one unified class: {sorted(duplicated)}")

    excluded = sorted(DACL10K_EXCLUDED_FROM_DAMAGE)
    actually_excluded = sorted(set(DACL10K_CLASSES) - set(mapped))
    if excluded != actually_excluded:
        raise RuntimeError(
            "Excluded DACL10K labels do not match DACL10K_EXCLUDED_FROM_DAMAGE: "
            f"declared {excluded}, derived {actually_excluded}"
        )

    if len(mapped) + len(excluded) != len(DACL10K_CLASSES):
        raise RuntimeError("The 19 -> 6 mapping does not cover every DACL10K class exactly once.")

    if list(groups) != UNIFIED_DAMAGE_CLASSES:
        raise RuntimeError("Unified channel order does not follow UNIFIED_DAMAGE_CLASSES.")
