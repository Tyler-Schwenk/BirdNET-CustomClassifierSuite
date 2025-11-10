"""
Common UI widgets and utilities for Streamlit interface.

This module provides reusable UI components that are used across multiple
tabs and panels (sweeps, hard negatives, file management, analysis).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st


def folder_picker(
    label: str,
    key: str,
    help_text: Optional[str] = None,
    initial_dir: Optional[Path] = None,
    relative_to: Optional[Path] = None,
    button_label: str = "📁 Browse",
    text_input: bool = True,
) -> Optional[str]:
    """
    Render a folder picker with native file dialog and optional text input fallback.
    
    This is a centralized folder picker used across the app for consistent behavior.
    
    Args:
        label: Display label for the folder picker
        key: Unique session state key for this picker
        help_text: Optional help text tooltip
        initial_dir: Initial directory to open file dialog in
        relative_to: If provided, convert absolute paths to relative from this location
        button_label: Label for the browse button (default: "📁 Browse")
        text_input: If True, also show a text input field for manual entry
        
    Returns:
        Selected folder path (relative if relative_to provided, else absolute), or None if not selected
        
    Example:
        >>> folder = folder_picker(
        ...     label="Select input folder",
        ...     key="my_input_folder",
        ...     relative_to=Path.cwd(),
        ...     help_text="Choose folder containing audio files"
        ... )
    """
    # Initialize session state for this picker
    session_key = f"_folder_picker_{key}"
    if session_key not in st.session_state:
        st.session_state[session_key] = ""
    
    # Render button
    col1, col2 = st.columns([4, 1])
    with col1:
        if label:
            st.markdown(f"**{label}**")
    with col2:
        if st.button(button_label, key=f"{key}_btn", help=help_text or "Select folder from file explorer"):
            selected = _open_folder_dialog(initial_dir or (relative_to if relative_to else Path.cwd()))
            if selected:
                # Convert to relative path if requested
                if relative_to:
                    try:
                        selected = selected.relative_to(relative_to)
                    except ValueError:
                        # Path is outside relative_to, keep absolute but warn
                        st.warning(f"⚠️ Selected folder is outside workspace: `{selected}`")
                
                st.session_state[session_key] = str(selected)
                st.rerun()
    
    # Render text input if enabled
    if text_input:
        result = st.text_input(
            "Or type path manually" if label else label,
            value=st.session_state[session_key],
            key=session_key,
            help=help_text,
            label_visibility="collapsed" if not label else "visible"
        )
        return result if result else None
    else:
        return st.session_state[session_key] if st.session_state[session_key] else None


def _open_folder_dialog(initial_dir: Path) -> Optional[Path]:
    """
    Open native OS folder picker dialog.
    
    Args:
        initial_dir: Directory to start the dialog in
        
    Returns:
        Selected Path or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create and hide root window
        root = tk.Tk()
        root.withdraw()
        
        # Keep dialog on top
        try:
            root.wm_attributes('-topmost', 1)
        except Exception:
            # Some platforms don't support this
            pass
        
        # Open folder picker
        folder_path = filedialog.askdirectory(
            title="Select Folder",
            initialdir=str(initial_dir) if initial_dir.exists() else os.getcwd()
        )
        
        root.destroy()
        
        if folder_path:
            return Path(folder_path)
        return None
        
    except ImportError:
        st.error("❌ tkinter not available. Please type folder paths manually.")
        return None
    except Exception as e:
        st.error(f"❌ Error opening folder picker: {e}")
        return None


def browse_folder(initial_dir: Optional[Path] = None, relative_to: Optional[Path] = None) -> Optional[str]:
    """
    Open a folder dialog and return the selected path.
    
    Public helper for getting a folder path without UI components.
    Use this when you want to open a folder dialog programmatically
    (e.g., inside a button click handler).
    
    Args:
        initial_dir: Directory to start browsing from
        relative_to: If provided, convert result to relative path from this location
        
    Returns:
        Selected folder path as string (relative if relative_to provided), or None
        
    Example:
        >>> if st.button("Browse"):
        ...     folder = browse_folder(Path.cwd(), relative_to=Path.cwd())
        ...     if folder:
        ...         st.write(f"Selected: {folder}")
    """
    selected_path = _open_folder_dialog(initial_dir or Path.cwd())
    
    if not selected_path:
        return None
    
    # Convert to relative path if requested
    if relative_to:
        try:
            selected_path = selected_path.relative_to(relative_to)
        except ValueError:
            # Path is outside relative_to
            st.warning(f"⚠️ Selected folder is outside workspace: `{selected_path}`")
    
    return str(selected_path)


def validate_folder_exists(folder_path: str | Path, show_message: bool = True) -> bool:
    """
    Validate that a folder path exists.
    
    Args:
        folder_path: Path to validate
        show_message: If True, show warning message in Streamlit
        
    Returns:
        True if folder exists and is a directory
        
    Example:
        >>> if validate_folder_exists("AudioData/input"):
        ...     process_files()
    """
    if not folder_path:
        return False
    
    path = Path(folder_path)
    if not path.exists():
        if show_message:
            st.warning(f"⚠️ Folder not found: `{folder_path}`")
        return False
    
    if not path.is_dir():
        if show_message:
            st.warning(f"⚠️ Path is not a directory: `{folder_path}`")
        return False
    
    return True


def validate_folder_not_empty(folder_path: str | Path, show_message: bool = True) -> bool:
    """
    Validate that a folder exists and contains files.
    
    Args:
        folder_path: Path to validate
        show_message: If True, show warning message in Streamlit
        
    Returns:
        True if folder exists and has content
        
    Example:
        >>> if validate_folder_not_empty("AudioData/input"):
        ...     count = len(list(Path("AudioData/input").iterdir()))
    """
    if not validate_folder_exists(folder_path, show_message):
        return False
    
    path = Path(folder_path)
    try:
        if not any(path.iterdir()):
            if show_message:
                st.warning(f"⚠️ Folder is empty: `{folder_path}`")
            return False
    except Exception as e:
        if show_message:
            st.warning(f"⚠️ Cannot read folder: {e}")
        return False
    
    return True


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string like "1.5 MB"
        
    Example:
        >>> format_file_size(1536000)
        '1.5 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
        
    Example:
        >>> format_duration(5025)
        '1h 23m 45s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def show_success_message(message: str, duration: int = 3) -> None:
    """
    Show a temporary success message that auto-dismisses.
    
    Args:
        message: Success message to display
        duration: Seconds to show message (default: 3)
    """
    placeholder = st.empty()
    placeholder.success(f"✓ {message}")
    import time
    time.sleep(duration)
    placeholder.empty()


def show_info_message(message: str, duration: int = 3) -> None:
    """
    Show a temporary info message that auto-dismisses.
    
    Args:
        message: Info message to display
        duration: Seconds to show message (default: 3)
    """
    placeholder = st.empty()
    placeholder.info(message)
    import time
    time.sleep(duration)
    placeholder.empty()


def confirm_action(message: str, key: str) -> bool:
    """
    Show a confirmation dialog before proceeding with an action.
    
    Args:
        message: Confirmation message
        key: Unique key for the confirmation state
        
    Returns:
        True if user confirmed
        
    Example:
        >>> if confirm_action("Delete all files?", "confirm_delete"):
        ...     delete_files()
    """
    confirm_key = f"_confirm_{key}"
    
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False
    
    if not st.session_state[confirm_key]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning(f"⚠️ {message}")
        with col2:
            if st.button("Confirm", key=f"{key}_confirm_btn"):
                st.session_state[confirm_key] = True
                st.rerun()
        return False
    else:
        # Reset confirmation state
        st.session_state[confirm_key] = False
        return True


def parse_number_list(text: str, cast_fn=float) -> list:
    """
    Parse comma-separated numbers from text.
    
    Args:
        text: Comma-separated number string (e.g., "1, 2.5, 3")
        cast_fn: Function to cast each value (default: float)
        
    Returns:
        List of parsed numbers
        
    Example:
        >>> parse_number_list("1, 2, 3", int)
        [1, 2, 3]
    """
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(cast_fn(token))
        except (ValueError, TypeError):
            pass
    return values


def parse_list_field(text: str) -> list[str]:
    """
    Parse newline or comma-separated text into list of strings.
    
    Args:
        text: Multi-line or comma-separated text
        
    Returns:
        List of non-empty strings
        
    Example:
        >>> parse_list_field("item1\\nitem2, item3")
        ['item1', 'item2', 'item3']
    """
    # Try newline-separated first
    if '\n' in text:
        items = [line.strip() for line in text.split('\n')]
    else:
        # Fall back to comma-separated
        items = [item.strip() for item in text.split(',')]
    
    return [item for item in items if item]


# ============================================================================
# Terminal/Command Output Widgets
# ============================================================================

def terminal_output(
    content: str,
    label: str = "Output",
    height: int = 400,
    max_lines: int = 1000,
    language: str = "text"
) -> None:
    """
    Display terminal/command output in a scrollable, copyable code block.
    
    This is the centralized terminal output widget used across the app for consistent
    display of command outputs, logs, and subprocess results.
    
    Args:
        content: Text content to display (will be truncated if too long)
        label: Label for the output block
        height: Height of the code block in pixels
        max_lines: Maximum number of lines to display (truncates from start if exceeded)
        language: Syntax highlighting language (default: "text", can be "python", "yaml", etc.)
        
    Example:
        >>> terminal_output(
        ...     content="\\n".join(log_lines),
        ...     label="Sweep execution log",
        ...     height=500,
        ...     max_lines=800
        ... )
    """
    lines = content.split('\n')
    if len(lines) > max_lines:
        # Keep most recent lines
        lines = lines[-max_lines:]
        display_content = '\n'.join(lines)
        st.caption(f"Showing last {max_lines} lines (truncated)")
    else:
        display_content = content
    
    st.code(display_content, language=language)


def stream_terminal_output(
    content: str,
    key: str,
    label: str = "Run log",
    height: int = 420,
    max_lines: int = 800
) -> None:
    """
    Display streaming terminal output in a disabled text area (auto-scrolls to bottom).
    
    Use this for live-updating command output where you need to update the content
    frequently. The disabled text area prevents user editing while remaining scrollable.
    
    Args:
        content: Current text content to display
        key: Unique Streamlit key for this widget
        label: Label for the text area
        height: Height in pixels
        max_lines: Maximum lines to keep (truncates from start)
        
    Example:
        >>> log_lines = []
        >>> log_area = st.empty()
        >>> for line in process.stdout:
        ...     log_lines.append(line)
        ...     with log_area:
        ...         stream_terminal_output(
        ...             content="\\n".join(log_lines),
        ...             key="sweep_log",
        ...             label="Live execution log"
        ...         )
    """
    lines = content.split('\n')
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    
    display_content = '\n'.join(lines)
    st.text_area(label, value=display_content, height=height, disabled=True, key=key)
