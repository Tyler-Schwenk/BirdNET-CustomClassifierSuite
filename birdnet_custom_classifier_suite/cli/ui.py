#!/usr/bin/env python3
"""
Launch Streamlit UI for BirdNET Custom Classifier Suite.

Provides interactive experiment analysis, sweep design, and result visualization.

Usage:
    birdnet-ui
    python -m birdnet_custom_classifier_suite.cli.ui
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def main():
    """Main entry point for launching Streamlit UI."""
    # Get the path to streamlit_app.py in the scripts directory
    # This assumes the package is installed or we're in development mode
    try:
        import birdnet_custom_classifier_suite
        pkg_root = Path(birdnet_custom_classifier_suite.__file__).parent.parent
        streamlit_app = pkg_root / "scripts" / "streamlit_app.py"
    except Exception:
        # Fallback: assume we're in the project root
        streamlit_app = Path("scripts/streamlit_app.py")
    
    if not streamlit_app.exists():
        print(f"ERROR: Could not find streamlit_app.py at {streamlit_app}")
        print("Make sure you're running from the project root or the package is installed.")
        sys.exit(1)
    
    print(f"Launching Streamlit UI from {streamlit_app}")
    
    # Launch streamlit
    try:
        subprocess.run(
            ["streamlit", "run", str(streamlit_app)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error launching Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: streamlit command not found. Make sure streamlit is installed:")
        print("    pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
