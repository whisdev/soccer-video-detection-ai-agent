#!/usr/bin/env python3
"""
Entry point for running soccer video detection.

Usage:
    python run.py --video path/to/video.mp4
    python run.py --video-dir videos/
"""

from pathlib import Path
import sys

# Ensure src and project root are on path
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from scripts.run_video import main

if __name__ == "__main__":
    main()
