#!/usr/bin/env python
"""Step 21 - map multilabel damage probabilities onto the bridge point cloud.

Thin wrapper around ``python -m damage3d`` (see damage3d/README.md).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from damage3d.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
