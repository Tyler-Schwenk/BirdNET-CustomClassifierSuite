# Folder Picker Feature - User Guide

## Overview

The UI now includes **folder picker buttons** (📁 Browse) that let you select subset folders using your native file explorer. This eliminates typos, validates paths automatically, and makes it easy to find the right folders.

## How to Use

### Step 1: Start the UI
```bash
streamlit run birdnet_custom_classifier_suite/ui/app.py
```

### Step 2: Navigate to Sweeps Tab
Click on the **Sweeps** tab in the sidebar.

### Step 3: Configure Data Composition Sweep Options
Scroll down to the **Data Composition Sweep Options** section.

### Step 4: Select Positive Subsets

#### Option A: Use File Explorer (Recommended)
1. Click the **📁 Browse** button next to "Positive Subsets"
2. Your file explorer opens
3. Navigate to `AudioData/curated/bestLowQuality/`
4. Select a folder (e.g., `small`, `medium`, `large`, or `top50`)
5. The folder path appears in the list below
6. Repeat to add more folders (each becomes a separate sweep combination)
7. Click ❌ next to any folder to remove it

#### Option B: Type Manually
You can also type or paste paths directly into the text area:
```
curated/bestLowQuality/small
curated/bestLowQuality/medium
curated/bestLowQuality/large
```

### Step 5: Select Negative Subsets

Same process:
1. Click the **📁 Browse** button next to "Negative Subsets"
2. Navigate to `AudioData/curated/hardNeg/`
3. Select folders (e.g., `hardneg_conf_min_85`, `hardneg_conf_min_99`)
4. Remove with ❌ if needed

### Step 6: Validation

The UI automatically validates all paths:
- ✅ **Green "All X folders validated"** = All paths exist
- ⚠️ **Yellow warning** = Some paths don't exist (lists which ones)

This prevents errors before you generate configs!

### Step 7: Generate Sweep

Click **Generate Sweep** button. The system:
1. Creates factorial combinations (e.g., 2 seeds × 3 positive × 2 negative = 12 configs)
2. Saves configs to `config/sweeps/<sweep_name>/`
3. Displays preview of generated configs

## UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  Data Composition Sweep Options                         │
│  Add curated positive and negative subset folders       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Positive Subsets                      📁 Browse        │
│  ───────────────────────────────────────────────────    │
│                                                          │
│  Selected positive subset folders:                      │
│  curated/bestLowQuality/small                      ❌   │
│  curated/bestLowQuality/medium                     ❌   │
│                                                          │
│  Or type paths manually:                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ curated/bestLowQuality/small                     │  │
│  │ curated/bestLowQuality/medium                    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  Negative Subsets                      📁 Browse        │
│  ───────────────────────────────────────────────────    │
│                                                          │
│  Selected negative subset folders:                      │
│  curated/hardNeg/hardneg_conf_min_85               ❌   │
│                                                          │
│  Or type paths manually:                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ curated/hardNeg/hardneg_conf_min_85              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ✓ All 3 subset folder(s) validated                    │
└─────────────────────────────────────────────────────────┘
```

## Features

### 1. Native File Explorer
- Uses Windows Explorer (or your OS's native file picker)
- Stays on top of other windows for easy selection
- Opens at workspace root by default

### 2. Relative Path Conversion
- Automatically converts `D:\...\AudioData\curated\...` to `AudioData/curated/...`
- Keeps paths portable across different machines
- Warns if you select a folder outside the workspace

### 3. Path Validation
- Checks every folder exists before allowing generation
- Visual feedback (green checkmark or yellow warning)
- Lists specific paths that are missing

### 4. Session Persistence
- Selected folders remain in the list until you remove them
- Text area syncs with the folder list automatically
- Can mix browse + manual entry

### 5. Easy Removal
- Click ❌ next to any folder to remove it
- No need to edit text manually

## Advanced Usage

### Multiple Folders Per Combination
You can combine multiple folders into a single sweep combination by typing comma-separated paths:

```
curated/bestLowQuality/small
curated/bestLowQuality/medium,curated/bestLowQuality/large
```

This creates 2 sweep combinations:
1. Just `small` folder
2. Both `medium` AND `large` folders (merged into training data)

### Empty Subsets
Leave both text areas empty if you don't want to sweep over subsets. The system will use your regular manifest-based data filtering (quality, balance, etc.) only.

## Troubleshooting

### "tkinter not available" Error
- tkinter should be included with Python on Windows
- If missing, you can still type paths manually in the text areas
- The text areas have the same functionality, just no file picker button

### "Selected folder is outside workspace" Warning
- You selected a folder that's not inside the workspace root
- The path will be absolute instead of relative
- This might cause issues if you move the workspace
- **Solution**: Navigate to folders inside the workspace (e.g., `AudioData/curated/...`)

### Paths Don't Validate
- Check for typos in manually entered paths
- Make sure folders actually exist (check in File Explorer)
- Use forward slashes `/` not backslashes `\`
- Paths are relative to workspace root: `AudioData/curated/...` not `./AudioData/...`

### Button Doesn't Open File Explorer
- Make sure you're clicking the 📁 Browse button (not the text area)
- On first click, it may take a second to open
- Check if the file picker opened behind other windows
- Try clicking "Generate Sweep" to refresh, then try Browse again

## Benefits Over Manual Entry

| Manual Entry | File Picker |
|--------------|-------------|
| Easy to make typos | No typos - select from actual folders |
| Hard to remember exact paths | Browse visually through folders |
| Need to check paths exist | Automatic validation |
| Need to type full relative path | Converted automatically |
| Must use correct slashes | Handled automatically |

## Example Workflow

**Goal**: Test 2 positive subsets and 2 negative subsets with 3 seeds = 12 experiments

1. Open UI Sweeps tab
2. Set seeds: `123, 456, 789`
3. Click 📁 Browse for Positive Subsets
   - Select `AudioData/curated/bestLowQuality/small` → Added
   - Click 📁 Browse again
   - Select `AudioData/curated/bestLowQuality/medium` → Added
4. Click 📁 Browse for Negative Subsets
   - Select `AudioData/curated/hardNeg/hardneg_conf_min_85` → Added
   - Click 📁 Browse again
   - Select `AudioData/curated/hardNeg/hardneg_conf_min_99` → Added
5. See validation: "✓ All 4 subset folder(s) validated"
6. See preview: "📊 This sweep will generate **12 configurations**"
7. Click "Generate Sweep"
8. Done! Configs created in `config/sweeps/<name>/`

The entire process takes ~30 seconds instead of several minutes of manual typing and path checking!
