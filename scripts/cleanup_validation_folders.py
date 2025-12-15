#!/usr/bin/env python3
"""
Script to delete 'Negative' and 'RADR' folders from validation packages in all experiments.

Usage:
    python scripts/cleanup_validation_folders.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path


def cleanup_validation_folders(experiments_dir: Path, dry_run: bool = False):
    """
    Delete 'Negative' and 'RADR' folders from validation packages.
    
    Args:
        experiments_dir: Path to the experiments directory
        dry_run: If True, only print what would be deleted without actually deleting
    """
    folders_to_delete = ["Negative", "RADR"]
    deleted_count = 0
    skipped_count = 0
    
    # Find all experiment directories
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return
    
    experiment_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
    
    print(f"Scanning {len(experiment_dirs)} experiment directories...")
    print(f"Looking for folders to delete: {', '.join(folders_to_delete)}")
    print(f"{'[DRY RUN] ' if dry_run else ''}Starting cleanup...\n")
    
    for exp_dir in experiment_dirs:
        validation_package = exp_dir / "validation_package"
        
        if not validation_package.exists():
            continue
        
        # Check for folders to delete
        for folder_name in folders_to_delete:
            folder_path = validation_package / folder_name
            
            if folder_path.exists() and folder_path.is_dir():
                if dry_run:
                    print(f"[DRY RUN] Would delete: {folder_path}")
                    deleted_count += 1
                else:
                    try:
                        shutil.rmtree(folder_path)
                        print(f"✓ Deleted: {folder_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"✗ Error deleting {folder_path}: {e}")
                        skipped_count += 1
    
    # Summary
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Folders {'that would be ' if dry_run else ''}deleted: {deleted_count}")
    if skipped_count > 0:
        print(f"  Folders skipped (errors): {skipped_count}")
    
    if dry_run:
        print("\nRun without --dry-run to actually delete the folders.")


def main():
    parser = argparse.ArgumentParser(
        description="Delete 'Negative' and 'RADR' folders from validation packages"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).parent.parent / "experiments",
        help="Path to experiments directory (default: ../experiments)"
    )
    
    args = parser.parse_args()
    
    cleanup_validation_folders(args.experiments_dir, args.dry_run)


if __name__ == "__main__":
    main()
