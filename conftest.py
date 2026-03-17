"""Root conftest.py to configure pytest for the entire project."""

import sys
from pathlib import Path

# Add src directory to Python path at module load time
# This must happen before any test imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
