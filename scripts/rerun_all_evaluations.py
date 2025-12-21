#!/usr/bin/env python
"""
Re-run evaluation on ALL experiments with the fixed evaluation code.

This script:
1. Checks all experiment directories
2. Backs up existing evaluation results  
3. Re-runs evaluation using the fixed load_results() that includes ALL files
4. Tracks which experiments changed (before/after metrics)
5. Generates a detailed log file for reporting

Usage:
    python scripts/rerun_all_evaluations.py
    python scripts/rerun_all_evaluations.py --stages stage11 stage12 stage14
    python scripts/rerun_all_evaluations.py --no-backup
"""

import argparse
import shutil
import json
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from birdnet_custom_classifier_suite.pipeline import evaluate_results


class ChangeTracker:
    """Track changes in metrics before/after re-evaluation."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.changes = []
        self.no_changes = []
        self.errors = []
        self.start_time = datetime.now()
        
    def read_metrics(self, exp_dir: Path) -> Optional[Dict]:
        """Read current metrics from experiment_summary.json."""
        summary_path = exp_dir / "evaluation" / "experiment_summary.json"
        if not summary_path.exists():
            return None
        
        try:
            with open(summary_path) as f:
                data = json.load(f)
            return {
                'test_ood_f1': data.get('metrics', {}).get('ood', {}).get('best_f1', {}).get('f1'),
                'test_ood_precision': data.get('metrics', {}).get('ood', {}).get('best_f1', {}).get('precision'),
                'test_ood_recall': data.get('metrics', {}).get('ood', {}).get('best_f1', {}).get('recall'),
                'test_iid_f1': data.get('metrics', {}).get('iid', {}).get('best_f1', {}).get('f1'),
                'timestamp': data.get('metadata', {}).get('timestamp'),
            }
        except Exception as e:
            print(f"  Warning: Could not read metrics: {e}")
            return None
    
    def record_change(self, exp_name: str, before: Optional[Dict], after: Optional[Dict]):
        """Record metrics change for an experiment."""
        if before is None:
            self.changes.append({
                'experiment': exp_name,
                'status': 'NEW',
                'before': None,
                'after': after,
            })
            return
        
        if after is None:
            self.errors.append({
                'experiment': exp_name,
                'error': 'Failed to generate new metrics',
            })
            return
        
        # Check if metrics changed significantly (>0.01 difference in F1)
        before_f1 = before.get('test_ood_f1', 0.0) or 0.0
        after_f1 = after.get('test_ood_f1', 0.0) or 0.0
        delta_f1 = after_f1 - before_f1
        
        if abs(delta_f1) > 0.01:
            self.changes.append({
                'experiment': exp_name,
                'status': 'CHANGED',
                'before': before,
                'after': after,
                'delta_f1': delta_f1,
            })
        else:
            self.no_changes.append({
                'experiment': exp_name,
                'before': before,
                'after': after,
            })
    
    def write_log(self):
        """Write comprehensive log file."""
        with open(self.log_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION RE-RUN LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {datetime.now() - self.start_time}\n")
            f.write("\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total experiments processed: {len(self.changes) + len(self.no_changes) + len(self.errors)}\n")
            f.write(f"  Changed (ΔF1 > 0.01): {len(self.changes)}\n")
            f.write(f"  Unchanged (ΔF1 ≤ 0.01): {len(self.no_changes)}\n")
            f.write(f"  Errors: {len(self.errors)}\n")
            f.write("\n")
            
            # Changed experiments (most important for report)
            if self.changes:
                f.write("="*80 + "\n")
                f.write("CHANGED EXPERIMENTS (METRICS CORRECTED)\n")
                f.write("="*80 + "\n")
                f.write("These experiments had significant metric changes and should be updated in reports.\n\n")
                
                # Sort by absolute delta F1 (biggest changes first)
                sorted_changes = sorted(self.changes, key=lambda x: abs(x.get('delta_f1', 0.0)), reverse=True)
                
                for change in sorted_changes:
                    exp = change['experiment']
                    before = change.get('before', {}) or {}
                    after = change.get('after', {}) or {}
                    delta = change.get('delta_f1', 0.0)
                    
                    f.write(f"\n{exp}\n")
                    f.write(f"  ΔF1: {delta:+.3f} ({'IMPROVED' if delta > 0 else 'DECREASED'})\n")
                    f.write(f"  BEFORE: F1={before.get('test_ood_f1', 'N/A'):.3f if before.get('test_ood_f1') else 'N/A'}, "
                           f"Precision={before.get('test_ood_precision', 'N/A'):.3f if before.get('test_ood_precision') else 'N/A'}, "
                           f"Recall={before.get('test_ood_recall', 'N/A'):.3f if before.get('test_ood_recall') else 'N/A'}\n")
                    f.write(f"  AFTER:  F1={after.get('test_ood_f1', 'N/A'):.3f if after.get('test_ood_f1') else 'N/A'}, "
                           f"Precision={after.get('test_ood_precision', 'N/A'):.3f if after.get('test_ood_precision') else 'N/A'}, "
                           f"Recall={after.get('test_ood_recall', 'N/A'):.3f if after.get('test_ood_recall') else 'N/A'}\n")
                
                f.write("\n")
            
            # Unchanged experiments
            if self.no_changes:
                f.write("="*80 + "\n")
                f.write("UNCHANGED EXPERIMENTS (NO CORRECTION NEEDED)\n")
                f.write("="*80 + "\n")
                f.write(f"These {len(self.no_changes)} experiments had minimal metric changes (ΔF1 ≤ 0.01).\n")
                f.write("No report updates needed for these.\n\n")
                
                for item in self.no_changes:
                    exp = item['experiment']
                    after = item.get('after', {}) or {}
                    f.write(f"  {exp}: F1={after.get('test_ood_f1', 'N/A'):.3f if after.get('test_ood_f1') else 'N/A'}\n")
                
                f.write("\n")
            
            # Errors
            if self.errors:
                f.write("="*80 + "\n")
                f.write("ERRORS\n")
                f.write("="*80 + "\n")
                for error in self.errors:
                    f.write(f"{error['experiment']}: {error['error']}\n")
                f.write("\n")


def backup_evaluation(exp_dir: Path):
    """Backup existing evaluation results."""
    eval_dir = exp_dir / "evaluation"
    if not eval_dir.exists():
        return False
    
    backup_dir = exp_dir / "evaluation_backup_old"
    if backup_dir.exists():
        return False  # Already backed up
    
    shutil.copytree(eval_dir, backup_dir)
    return True


def rerun_evaluation_with_tracking(exp_dir: Path, tracker: ChangeTracker, backup: bool = True) -> bool:
    """Re-run evaluation for a single experiment and track changes."""
    exp_name = exp_dir.name
    
    # Read metrics before re-evaluation
    metrics_before = tracker.read_metrics(exp_dir)
    
    # Check if experiment has inference results
    inference_dir = exp_dir / "inference"
    if not inference_dir.exists():
        print(f"  [SKIP] {exp_name}: No inference directory, skipping")
        return False
    
    # Backup existing results
    if backup:
        if backup_evaluation(exp_dir):
            print(f"  [BACKUP] {exp_name}: Backed up old evaluation")
    
    # Run evaluation
    try:
        evaluate_results.run_evaluation(str(exp_dir))
        
        # Read metrics after re-evaluation
        metrics_after = tracker.read_metrics(exp_dir)
        
        # Track the change
        tracker.record_change(exp_name, metrics_before, metrics_after)
        
        # Print status
        if metrics_before and metrics_after:
            before_f1 = metrics_before.get('test_ood_f1', 0.0) or 0.0
            after_f1 = metrics_after.get('test_ood_f1', 0.0) or 0.0
            delta = after_f1 - before_f1
            
            if abs(delta) > 0.01:
                print(f"  [OK] {exp_name}: F1 {before_f1:.3f} -> {after_f1:.3f} (Delta={delta:+.3f})")
            else:
                print(f"  [OK] {exp_name}: F1 {after_f1:.3f} (no change)")
        else:
            print(f"  [OK] {exp_name}: Evaluation complete")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {exp_name}: {e}")
        tracker.errors.append({
            'experiment': exp_name,
            'error': str(e),
        })
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Re-run evaluation on ALL experiments with fixed code and track changes"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        help="Specific stages to re-evaluate (e.g., stage11 stage12). If not provided, all stages are processed."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up existing evaluations"
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Base experiments directory"
    )
    parser.add_argument(
        "--log-file",
        default="evaluation_rerun_log.txt",
        help="Path to output log file"
    )
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    log_path = Path(args.log_file)
    backup = not args.no_backup
    
    # Initialize tracker
    tracker = ChangeTracker(log_path)
    
    # Find all experiment directories
    if args.stages:
        # Process specific stages
        exp_paths = []
        for stage in args.stages:
            matches = sorted(experiments_dir.glob(f"{stage}_*"))
            exp_paths.extend(matches)
        
        if not exp_paths:
            print(f"ERROR: No experiments found for stages: {args.stages}")
            return
    else:
        # Process all stage* experiments
        exp_paths = sorted(experiments_dir.glob("stage*_*"))
    
    print(f"\n{'='*80}")
    print(f"RE-RUNNING EVALUATION FOR {len(exp_paths)} EXPERIMENTS")
    print(f"{'='*80}\n")
    print(f"Log file: {log_path.absolute()}")
    print(f"Backup old evaluations: {backup}")
    print()
    
    # Process each experiment
    success_count = 0
    for exp_dir in exp_paths:
        if rerun_evaluation_with_tracking(exp_dir, tracker, backup):
            success_count += 1
    
    # Write log file
    print(f"\n{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully re-evaluated: {success_count}/{len(exp_paths)}")
    print(f"Experiments with changed metrics: {len(tracker.changes)}")
    print(f"Experiments unchanged: {len(tracker.no_changes)}")
    print(f"Errors: {len(tracker.errors)}")
    print()
    
    tracker.write_log()
    print(f"[OK] Detailed log written to: {log_path.absolute()}")
    print()
    print("Next steps:")
    print("  1. Review the log file to see which experiments changed")
    print("  2. Update your report with the corrected metrics")
    print("  3. Run: python -m birdnet_custom_classifier_suite.pipeline.collect_experiments")
    print("  4. Update all_experiments.csv in your analysis")


if __name__ == "__main__":
    main()
