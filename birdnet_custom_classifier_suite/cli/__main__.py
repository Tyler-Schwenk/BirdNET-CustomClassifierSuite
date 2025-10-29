"""
CLI entry point for running subcommands.

Allows execution via: python -m birdnet_custom_classifier_suite.cli <command>
"""

from __future__ import annotations

import sys


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        print("BirdNET Custom Classifier Suite CLI")
        print("\nAvailable commands:")
        print("  analyze    - Analyze experiment results and generate leaderboards")
        print("  ui         - Launch interactive Streamlit UI")
        print("\nUsage:")
        print("  python -m birdnet_custom_classifier_suite.cli analyze [options]")
        print("  python -m birdnet_custom_classifier_suite.cli ui")
        print("\nOr use the installed console scripts:")
        print("  birdnet-analyze [options]")
        print("  birdnet-ui")
        sys.exit(1)
    
    command = sys.argv[1]
    # Remove the command from argv so subcommands see clean arguments
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "analyze":
        from .analyze import main as analyze_main
        analyze_main()
    elif command == "ui":
        from .ui import main as ui_main
        ui_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: analyze, ui")
        sys.exit(1)


if __name__ == "__main__":
    main()
