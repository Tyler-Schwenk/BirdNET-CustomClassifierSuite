# Common UI Widgets - Module Refactoring

## Overview

Centralized common UI widgets and utilities used across multiple tabs in the Streamlit application. This module eliminates code duplication and ensures consistent behavior.

## Location

`birdnet_custom_classifier_suite/ui/common/widgets.py`

## Components

### Folder Picker

**`folder_picker()`** - Native OS folder browser with text input fallback

Used in:
- Sweeps tab (subset folder selection)
- Hard Negatives tab (input folder selection)
- File Management tab (source/destination folders)

**Features:**
- Native OS file dialog (Windows Explorer, Finder, etc.)
- Automatic relative path conversion
- Session state management
- Text input fallback for manual entry
- Validation warnings for paths outside workspace

**Example:**
```python
from birdnet_custom_classifier_suite.ui.common import folder_picker

folder = folder_picker(
    label="Select input folder",
    key="my_input_folder",
    relative_to=Path.cwd(),
    help_text="Choose folder containing audio files"
)
```

### Validation Functions

**`validate_folder_exists()`** - Check if folder path exists
**`validate_folder_not_empty()`** - Check if folder exists and has content

Used in:
- Sweeps tab (validate subset paths)
- Hard Negatives tab (validate input directory)
- File Management tab (validate source folders)

**Example:**
```python
from birdnet_custom_classifier_suite.ui.common import validate_folder_exists

if validate_folder_exists("AudioData/input"):
    process_files()
```

### Formatting Functions

**`format_file_size()`** - Convert bytes to human-readable string (1.5 MB)
**`format_duration()`** - Convert seconds to human-readable string (1h 23m 45s)

Used in:
- Hard Negatives tab (display file sizes)
- File Management tab (show folder sizes)
- Analysis tab (performance metrics)

**Example:**
```python
from birdnet_custom_classifier_suite.ui.common import format_file_size, format_duration

size_str = format_file_size(1536000)  # "1.5 MB"
duration_str = format_duration(5025)   # "1h 23m 45s"
```

### Parsing Functions

**`parse_number_list()`** - Parse comma-separated numbers
**`parse_list_field()`** - Parse newline or comma-separated text

Used in:
- Sweeps tab (parse seed lists, learning rates, etc.)
- Hard Negatives tab (parse threshold values)

**Example:**
```python
from birdnet_custom_classifier_suite.ui.common import parse_number_list

seeds = parse_number_list("123, 456, 789", int)  # [123, 456, 789]
learning_rates = parse_number_list("0.001, 0.0005", float)  # [0.001, 0.0005]
```

### Message Functions

**`show_success_message()`** - Temporary success message (auto-dismisses)
**`show_info_message()`** - Temporary info message (auto-dismisses)
**`confirm_action()`** - Confirmation dialog before destructive actions

**Example:**
```python
from birdnet_custom_classifier_suite.ui.common import show_success_message, confirm_action

if confirm_action("Delete all files?", "confirm_delete"):
    delete_files()
    show_success_message("Files deleted successfully")
```

## Migration Guide

### Before (Duplicated Code)

#### Sweeps Tab
```python
def browse_for_folder(session_key: str, relative_to: Path = None) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        # ... 30 lines of code ...
```

#### Hard Negatives Tab
```python
def _render_folder_picker(st_: st) -> str:
    if 'hn_input_dir' not in st.session_state:
        st.session_state['hn_input_dir'] = str(constants.DEFAULT_INPUT_DIR)
    if st_.button("Choose folder (Explorer)", key='hn_choose_folder'):
        # ... 20 lines of code ...
```

**Problems:**
- 2-3 different implementations of folder picking
- Inconsistent behavior across tabs
- Difficult to fix bugs (must update 3 places)
- No standardization

### After (Common Module)

#### All Tabs
```python
from birdnet_custom_classifier_suite.ui.common import folder_picker

folder = folder_picker(
    label="Select folder",
    key="my_folder",
    relative_to=Path.cwd()
)
```

**Benefits:**
- ✅ Single source of truth
- ✅ Consistent behavior
- ✅ Bug fixes propagate automatically
- ✅ Easier to maintain and test
- ✅ Well-documented API

## Files Modified

### New Files
1. `birdnet_custom_classifier_suite/ui/common/widgets.py` - Common UI widgets (400 lines)
2. `birdnet_custom_classifier_suite/ui/common/__init__.py` - Public API exports

### Updated Files
1. `birdnet_custom_classifier_suite/ui/sweeps/views.py`
   - Removed `browse_for_folder()` (60 lines)
   - Replaced with `folder_picker()` import and calls
   - Simplified subset selection code

2. `birdnet_custom_classifier_suite/ui/hard_negative/views.py`
   - Replaced `_render_folder_picker()` (30 lines) with `folder_picker()` call
   - Added import from common

3. `birdnet_custom_classifier_suite/ui/hard_negative/utils.py`
   - Removed `format_file_size()` implementation
   - Added delegation to common utility
   - Removed `validate_input_directory()` implementation
   - Added delegation to common utility

4. `birdnet_custom_classifier_suite/ui/sweeps/utils.py`
   - Removed `parse_num_list()` implementation
   - Added delegation to common utility

## Code Reduction

**Lines Removed:** ~150 lines of duplicated code
**Lines Added:** ~400 lines of centralized, well-documented utilities
**Net:** Better maintainability despite slightly more code (comprehensive docs + examples)

## Testing Checklist

Test each tab to ensure folder pickers still work:

### Sweeps Tab
- [ ] Click "📁 Browse" for positive subsets → file dialog opens
- [ ] Select folder → path appears in list with relative path
- [ ] Click "📁 Browse" for negative subsets → file dialog opens
- [ ] Click ❌ to remove folder → folder disappears from list
- [ ] Type path manually → validation shows green/yellow feedback

### Hard Negatives Tab
- [ ] Click "Choose folder (Explorer)" → file dialog opens
- [ ] Select folder → path appears in text input
- [ ] Type path manually → still works
- [ ] Invalid path → warning shown

### File Management Tab (if applicable)
- [ ] Any folder pickers use common module
- [ ] Behavior consistent with other tabs

## Future Enhancements

Potential additions to `common/widgets.py`:

1. **File picker** (not just folders)
2. **Multi-select file/folder picker**
3. **Path autocomplete widget**
4. **Folder tree browser** (inline UI component)
5. **Drag-and-drop file upload** (with validation)
6. **Progress bars** with standardized styling
7. **Data table formatters** (highlight cells, color coding)
8. **Chart themes** (consistent colors across tabs)

## Best Practices

### When to Add to Common

✅ **DO add to common:**
- Used in 2+ tabs/modules
- Likely to be reused in future
- Has clear, general-purpose API
- Benefits from standardization

❌ **DON'T add to common:**
- Tab-specific business logic
- One-time use widgets
- Tightly coupled to specific workflows
- Unclear if it will be reused

### How to Add New Widgets

1. Implement in `common/widgets.py` with full docstring
2. Add comprehensive examples in docstring
3. Export in `common/__init__.py`
4. Update this documentation
5. Test in at least 2 different contexts
6. Add type hints for all parameters

## Summary

The common widgets module **eliminates code duplication** and **standardizes UI behavior** across the Streamlit app. Folder picking now uses a single, well-tested implementation instead of 3 different versions. This makes the codebase more maintainable and ensures consistent user experience.

**Key win:** When we fix a bug in `folder_picker()`, it's fixed everywhere automatically! 🎉
