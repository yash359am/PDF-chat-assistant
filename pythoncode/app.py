"""
Launcher wrapper for pythoncode/app.py
Delegates execution to root app.py using runpy to prevent module caching across reruns.
"""

import sys
import runpy
from pathlib import Path

root_app = Path(__file__).parent.parent / "app.py"
runpy.run_path(str(root_app), run_name="__main__")