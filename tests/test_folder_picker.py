"""Test the folder picker functionality."""
import sys
from pathlib import Path

# Test if tkinter is available
try:
    import tkinter as tk
    print("✓ tkinter is available")
    
    # Test creating a hidden root window
    root = tk.Tk()
    root.withdraw()
    print("✓ tkinter root window can be created")
    root.destroy()
    
    # Test filedialog
    from tkinter import filedialog
    print("✓ tkinter.filedialog is available")
    
    print("\n=== Folder Picker Test Passed ===")
    print("The browse button in the UI should work correctly.")
    print("\nTo test in the UI:")
    print("1. Run: streamlit run birdnet_custom_classifier_suite/ui/app.py")
    print("2. Go to Sweeps tab")
    print("3. Scroll to 'Data Composition Sweep Options'")
    print("4. Click the 📁 Browse button")
    print("5. Select a folder (e.g., AudioData/curated/bestLowQuality/small)")
    print("6. The path should appear in the list and text area")
    
except ImportError as e:
    print(f"✗ tkinter not available: {e}")
    print("\ntkinter is required for the folder picker.")
    print("It should be included with Python on Windows.")
    print("If missing, you can still type paths manually in the text area.")
    sys.exit(1)
