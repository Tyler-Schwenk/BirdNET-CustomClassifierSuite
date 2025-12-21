#!/usr/bin/env python
"""
Analyze which experiments had metric changes after re-evaluation.
Compares current metrics to backed up metrics.
"""

from pathlib import Path
import json
from collections import defaultdict

def read_metrics(json_path):
    """Read metrics from experiment_summary.json."""
    if not json_path.exists():
        return None
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        return {
            'ood_f1': data.get('metrics', {}).get('ood', {}).get('best_f1', {}).get('f1'),
            'ood_precision': data.get('metrics', {}).get('ood', {}).get('best_f1', {}).get('precision'),
            'ood_recall': data.get('metrics', {}).get('ood', {}).get('best_f1', {}).get('recall'),
            'iid_f1': data.get('metrics', {}).get('iid', {}).get('best_f1', {}).get('f1'),
        }
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None


def extract_stage(exp_name):
    """Extract stage name from experiment name."""
    import re
    match = re.match(r'(stage\d+[a-z]*)', exp_name)
    return match.group(1) if match else 'unknown'


def main():
    experiments_dir = Path('experiments')
    
    # Find all experiments with backups (means they were re-evaluated)
    exp_dirs = sorted(experiments_dir.glob('stage*_*'))
    
    changes_by_stage = defaultdict(list)
    no_change_by_stage = defaultdict(list)
    no_backup_by_stage = defaultdict(list)
    
    print("Analyzing evaluation changes...\n")
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        stage = extract_stage(exp_name)
        
        # Read current metrics
        current_json = exp_dir / 'evaluation' / 'experiment_summary.json'
        current = read_metrics(current_json)
        
        # Read backup metrics
        backup_json = exp_dir / 'evaluation_backup_old' / 'experiment_summary.json'
        backup = read_metrics(backup_json)
        
        if backup is None:
            no_backup_by_stage[stage].append(exp_name)
            continue
        
        if current is None:
            print(f"WARNING: {exp_name} has backup but no current metrics")
            continue
        
        # Calculate change
        before_f1 = backup.get('ood_f1', 0.0) or 0.0
        after_f1 = current.get('ood_f1', 0.0) or 0.0
        delta = after_f1 - before_f1
        
        if abs(delta) > 0.01:
            changes_by_stage[stage].append({
                'name': exp_name,
                'before_f1': before_f1,
                'after_f1': after_f1,
                'delta': delta,
            })
        else:
            no_change_by_stage[stage].append({
                'name': exp_name,
                'f1': after_f1,
            })
    
    # Write summary report
    output_path = Path('results/evaluation_changes_by_stage.txt')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION CHANGES SUMMARY - BY STAGE\n")
        f.write("="*80 + "\n\n")
        
        f.write("PURPOSE: Use this to determine which stages need report updates.\n")
        f.write("- Stages with CHANGES: Update your report with corrected metrics\n")
        f.write("- Stages with NO CHANGES: Keep existing report data\n\n")
        
        # Summary by stage
        all_stages = sorted(set(list(changes_by_stage.keys()) + 
                                list(no_change_by_stage.keys()) + 
                                list(no_backup_by_stage.keys())))
        
        f.write("="*80 + "\n")
        f.write("QUICK SUMMARY BY STAGE\n")
        f.write("="*80 + "\n\n")
        
        for stage in all_stages:
            changed = len(changes_by_stage.get(stage, []))
            unchanged = len(no_change_by_stage.get(stage, []))
            no_backup = len(no_backup_by_stage.get(stage, []))
            total = changed + unchanged + no_backup
            
            status = "CHANGED" if changed > 0 else "UNCHANGED"
            
            f.write(f"{stage}: {status}\n")
            f.write(f"  Total: {total} experiments\n")
            f.write(f"  Changed: {changed}\n")
            f.write(f"  Unchanged: {unchanged}\n")
            if no_backup > 0:
                f.write(f"  Not re-evaluated: {no_backup}\n")
            
            if changed > 0:
                f.write(f"  ACTION REQUIRED: Update report for {stage}\n")
            else:
                f.write(f"  No action needed - metrics already correct\n")
            
            f.write("\n")
        
        # Detailed changes by stage
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED CHANGES BY STAGE\n")
        f.write("="*80 + "\n\n")
        
        for stage in all_stages:
            changed = changes_by_stage.get(stage, [])
            
            if not changed:
                continue
            
            f.write(f"\n{stage.upper()} - {len(changed)} EXPERIMENTS CHANGED\n")
            f.write("-"*80 + "\n")
            
            # Sort by absolute delta (biggest changes first)
            changed_sorted = sorted(changed, key=lambda x: abs(x['delta']), reverse=True)
            
            for item in changed_sorted:
                f.write(f"\n  {item['name']}\n")
                f.write(f"    BEFORE: F1={item['before_f1']:.3f}\n")
                f.write(f"    AFTER:  F1={item['after_f1']:.3f}\n")
                f.write(f"    DELTA:  {item['delta']:+.3f} ({'IMPROVED' if item['delta'] > 0 else 'DECREASED'})\n")
    
    print(f"\n[OK] Analysis complete!")
    print(f"Summary written to: {output_path.absolute()}\n")
    
    # Print quick summary to console
    print("="*80)
    print("QUICK SUMMARY - STAGES REQUIRING REPORT UPDATES")
    print("="*80 + "\n")
    
    stages_with_changes = []
    stages_without_changes = []
    
    for stage in all_stages:
        changed = len(changes_by_stage.get(stage, []))
        if changed > 0:
            stages_with_changes.append((stage, changed))
        else:
            stages_without_changes.append(stage)
    
    if stages_with_changes:
        print("STAGES WITH CHANGES (Update your report):")
        for stage, count in stages_with_changes:
            print(f"  - {stage}: {count} experiments changed")
    else:
        print("No stages had metric changes!")
    
    print()
    
    if stages_without_changes:
        print("STAGES WITHOUT CHANGES (No report updates needed):")
        for stage in stages_without_changes:
            unchanged = len(no_change_by_stage.get(stage, []))
            print(f"  - {stage}: {unchanged} experiments (already correct)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
