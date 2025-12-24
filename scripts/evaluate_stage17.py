#!/usr/bin/env python3
"""
Evaluate all stage17 experiments and collect results into all_experiments.csv
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from birdnet_custom_classifier_suite.pipeline import evaluate_results, collect_experiments

def main():
    experiments_root = Path("experiments")
    
    # Find all stage17 experiments
    stage17_dirs = sorted([d for d in experiments_root.iterdir() 
                          if d.is_dir() and d.name.startswith("stage17_")])
    
    print(f"Found {len(stage17_dirs)} stage17 experiments")
    
    # Evaluate each one
    for i, exp_dir in enumerate(stage17_dirs, 1):
        print(f"\n[{i}/{len(stage17_dirs)}] Evaluating {exp_dir.name}...")
        
        # Check if results already exist
        results_file = exp_dir / "results" / "results.csv"
        if results_file.exists():
            print(f"  ✓ Results already exist, skipping")
            continue
        
        try:
            evaluate_results.run_evaluation(str(exp_dir))
            print(f"  ✓ Evaluation complete")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Collect all experiments into master CSV
    print("\n" + "="*60)
    print("Collecting all experiments into all_experiments.csv...")
    collect_experiments.collect_experiments("experiments", "results/all_experiments.csv")
    print("Done!")

if __name__ == "__main__":
    main()
