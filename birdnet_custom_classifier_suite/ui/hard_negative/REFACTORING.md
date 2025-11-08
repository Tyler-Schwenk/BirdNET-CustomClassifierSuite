# Hard-Negative Mining Module Refactoring

## Overview
The hard-negative mining UI module has been fully refactored following best practices for code organization, maintainability, and usability.

## Module Structure

### Files

#### `__init__.py`
Module initialization that exports the main panel function.
```python
from birdnet_custom_classifier_suite.ui.hard_negative.views import hard_negative_panel
```

#### `constants.py`
Centralized configuration and magic values:
- **Path Constants**: `DEFAULT_INPUT_DIR`, `DEFAULT_OUTPUT_ROOT`, `EXPERIMENTS_ROOT`, `RESULTS_DIR_NAME`
- **Model Configuration**: `MODEL_EXTENSIONS`, `DEFAULT_SENSITIVITY`, `DEFAULT_MIN_CONF`
- **UI Settings**: `UI_UPDATE_INTERVAL`, `MAX_LOG_LINES`, `MAX_PREVIEW_FILES`
- **Selection Modes**: `SELECTION_MODE_TOP_K`, `SELECTION_MODE_PERCENTILE`, `SELECTION_MODE_THRESHOLD`, `SELECTION_MODE_MANUAL`
- **Export Methods**: `EXPORT_METHOD_COPY`, `EXPORT_METHOD_HARDLINK`, `EXPORT_METHOD_SYMLINK`
- **Default Values**: Various defaults for UI inputs

#### `utils.py`
Common utility functions extracted from views:
- `get_output_path_for_model_source()` - Determine save paths based on model source
- `save_results_csv()` - Save aggregated results with metadata
- `copy_files()` - File copying with error handling
- `write_selection_report()` - Export selection metadata as JSON
- `validate_input_directory()` - Directory validation
- `parse_csv_columns()` - CSV column detection
- `format_file_size()` - Human-readable file size formatting
- `get_experiment_model_files()` - Find model files in experiment directories

#### `views.py`
Main Streamlit UI panel, refactored into focused helper functions:
- `_render_folder_picker()` - Native folder dialog + text input
- `_get_experiment_list()` - Cached experiment discovery
- `_render_model_selection()` - Model source selection UI
- `_stream_analyzer_logs()` - Real-time process output display
- `_run_inference_workflow()` - Complete inference execution
- `_load_csv_workflow()` - CSV upload and file matching
- `_render_selection_ui()` - File selection controls (top-k, percentile, threshold, manual)
- `_render_preview()` - Audio preview widget
- `_copy_files()` - Export selected files
- `hard_negative_panel()` - Main entry point

#### `engine.py`
Business logic for analyzer invocation:
- `run_analyzer_cli()` - Blocking analyzer execution
- `run_analyzer_cli_stream()` - Streaming analyzer execution (returns process handle)
- `run_inference_for_experiment()` - Experiment-based inference
- `run_inference_for_experiment_stream()` - Streaming experiment inference
- `run_inference_and_collect()` - Run and aggregate results
- `run_inference_and_collect_stream()` - Streaming run and collect
- `collect_per_file_max()` - Aggregate CSV outputs to per-file DataFrame

#### `curator.py`
File operations and CSV aggregation:
- `make_links()` - Create hard/symlinks or copy files
- `match_files()` - Optimized basename matching between DataFrame and filesystem
- `aggregate_results()` - Combine multiple CSV files into master
- `load_radr_csv()` - Load and validate RADR CSV format

#### `aggregate_and_curate.py`
CLI script for batch operations (kept for backward compatibility):
- Can be run as: `python -m birdnet_custom_classifier_suite.ui.hard_negative.aggregate_and_curate`
- Provides command-line interface for aggregation and curation

## Key Improvements

### 1. Code Organization
- **Single Responsibility**: Each function has one clear purpose
- **Separation of Concerns**: UI (views), business logic (engine), file ops (curator), config (constants), utilities (utils)
- **DRY Principle**: Eliminated code duplication through extraction

### 2. Maintainability
- **Constants Centralization**: All magic values in one place for easy modification
- **Type Hints**: Comprehensive type annotations for better IDE support and documentation
- **Docstrings**: Clear documentation for all public functions
- **Error Handling**: Improved error messages and fallback behavior

### 3. User Experience
- **Emojis**: Visual cues for better UI navigation
- **Progress Feedback**: Real-time streaming output during long operations
- **Persistent Results**: Output saved to accessible folders instead of temp directories
- **Native Dialogs**: Windows Explorer integration for folder selection

### 4. Code Quality
- **Line Length**: Functions kept under 100 lines where possible
- **Nesting Depth**: Reduced complexity through early returns and extraction
- **Imports**: Clean, organized imports at module level
- **No Syntax Errors**: All files pass static analysis

## Migration Guide

### For Developers

#### Accessing Constants
```python
# OLD
DEFAULT_INPUT = Path("scripts") / "input"
EXPERIMENTS_ROOT = Path('experiments')

# NEW
from birdnet_custom_classifier_suite.ui.hard_negative import constants
input_dir = constants.DEFAULT_INPUT_DIR
exp_root = constants.EXPERIMENTS_ROOT
```

#### Using Utilities
```python
# OLD (inline code)
def some_function():
    if model_source == "Use experiment...":
        return Path(...), label, model
    # ... many lines ...

# NEW (utility function)
from birdnet_custom_classifier_suite.ui.hard_negative import utils
path, label, model = utils.get_output_path_for_model_source(
    model_source, exp, model_choice, uploaded
)
```

#### Streaming Analyzer Output
```python
# OLD (blocking)
out_dir, cmd = engine.run_analyzer_cli(args)

# NEW (streaming)
proc, out_dir, cmd = engine.run_analyzer_cli_stream(args)
for line in proc.stdout:
    # Process line-by-line for real-time feedback
    print(line)
proc.wait()
```

### For Users

No changes to UI functionality - all workflows remain the same:
1. **Run Inference**: Select folder → Choose model → Run analyzer → View results
2. **Load CSV**: Upload existing results → Match to files → Select candidates
3. **Preview**: Listen to audio samples before export
4. **Export**: Choose selection mode → Export to curated folder

## Testing Checklist

### Critical Workflows
- [ ] Run inference using experiment (canonical args)
- [ ] Run inference using experiment model file
- [ ] Run inference using uploaded model
- [ ] Load existing RADR CSV and match files
- [ ] Select files using top-k mode
- [ ] Select files using percentile mode
- [ ] Select files using threshold mode
- [ ] Manual selection via checkboxes
- [ ] Preview audio files
- [ ] Export as copy
- [ ] Export as hardlink
- [ ] Export as symlink
- [ ] Aggregate multiple result CSVs

### Edge Cases
- [ ] Empty input folder
- [ ] Non-existent folder path
- [ ] CSV with no matching files
- [ ] Model file selection with no models available
- [ ] Analyzer crash/error during execution
- [ ] Permissions error during file export

### UI Validation
- [ ] Folder picker dialog opens correctly
- [ ] Experiment list refreshes properly
- [ ] Model file dropdown populates correctly
- [ ] Progress bar updates during analysis
- [ ] Log output streams in real-time
- [ ] Results CSV is saved to correct location
- [ ] Selection report JSON is created

## Performance Considerations

### Optimizations
- **Session State Caching**: Experiment list cached to avoid repeated filesystem scans
- **Streaming Output**: Non-blocking subprocess execution with line-buffered output
- **Lazy Loading**: Model files only listed when experiment selected
- **Efficient Matching**: Optimized basename matching using dictionaries

### Resource Usage
- **Memory**: Log lines limited to last 100 for display
- **Disk**: Results saved incrementally during analysis
- **CPU**: Analyzer runs in subprocess without blocking UI thread

## Future Enhancements

### Potential Improvements
1. **Async Processing**: Use asyncio for concurrent operations
2. **Progress Persistence**: Save progress to resume interrupted analyses
3. **Batch Operations**: Process multiple folders in sequence
4. **Advanced Filtering**: More sophisticated selection criteria
5. **Export Templates**: Predefined export configurations
6. **Result Comparison**: Compare multiple analysis runs side-by-side
7. **Visualization**: Charts/graphs for confidence distributions

### Architecture Considerations
- Consider splitting `engine.py` into `analyzer.py` and `aggregator.py` if it grows
- May want to create `models.py` or `types.py` for data classes if complexity increases
- Could extract shared file operations from `curator.py` into `file_utils.py`

## Dependencies

### Internal
- `birdnet_custom_classifier_suite.pipeline` - For building analyzer commands
- `birdnet_custom_classifier_suite.ui.common` - Shared UI components (if any)

### External
- `streamlit` - Web UI framework
- `pandas` - DataFrame operations
- `pathlib` - Path handling (stdlib)
- `subprocess` - Process execution (stdlib)
- `tkinter` - Native dialogs (stdlib)
- `shutil` - File operations (stdlib)

## Documentation

- See `QUICKSTART.md` for user guide
- See `docs/SWEEPS.md` for related sweep functionality
- API documentation auto-generated from docstrings
