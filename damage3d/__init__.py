"""damage3d - geometric pipeline that maps per-image multilabel damage
probabilities onto the Metashape point cloud of a bridge.

Stages: 2D probability maps -> Metashape cameras -> visible 3D points ->
multi-view fusion -> inspectable 3D outputs.

The deep-learning model is NOT part of this package. It is represented by a
``ProbabilityProvider`` (see ``damage3d.providers``); replacing the synthetic
provider with model outputs does not require changes to projection,
visibility, fusion or export.
"""

__version__ = "0.1.0"
