#!/usr/bin/env python3
"""
Verify Stage 6 sweep file selection and subset representation.
Shows exactly how many files came from manifest vs subsets.
"""

import pandas as pd
from pathlib import Path
import json

def analyze_experiment(exp_name: str):
    """Analyze file selection for one experiment."""
    exp_dir = Path(f"experiments/{exp_name}")
    
    if not exp_dir.exists():
        print(f"❌ {exp_name} not found")
        return None
    
    # Load selection report
    report_path = exp_dir / "training_package" / "selection_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    # Load selection manifest
    manifest_path = exp_dir / "training_package" / "selection_manifest.csv"
    if not manifest_path.exists():
        print(f"⚠️  {exp_name}: No selection_manifest.csv found")
        return report
    
    df = pd.read_csv(manifest_path)
    
    # Count by source
    pos = df[df['label'].str.lower() == 'positive']
    neg = df[df['label'].str.lower() == 'negative']
    
    # Identify source by path since source_subset column may not exist
    def identify_source(row):
        path = str(row['resolved_path'])
        if 'curated' in path.lower():
            # Extract subset name from path
            if 'hardNeg' in path:
                parts = path.split('hardNeg')
                if len(parts) > 1:
                    subset = parts[1].split('\\')[1] if '\\' in parts[1] else parts[1].split('/')[1]
                    return f"hardNeg/{subset}"
            elif 'bestLowQuality' in path:
                parts = path.split('bestLowQuality')
                if len(parts) > 1:
                    subset = parts[1].split('\\')[1] if '\\' in parts[1] else parts[1].split('/')[1]
                    return f"bestLowQuality/{subset}"
            return "curated (other)"
        else:
            return "manifest"
    
    pos['source'] = pos.apply(identify_source, axis=1)
    neg['source'] = neg.apply(identify_source, axis=1)
    
    pos_sources = pos.groupby('source').size().to_dict()
    neg_sources = neg.groupby('source').size().to_dict()
    
    return {
        'report': report,
        'pos_sources': pos_sources,
        'neg_sources': neg_sources,
        'pos_total': len(pos),
        'neg_total': len(neg)
    }


def print_experiment_analysis(exp_name: str):
    """Print detailed analysis for one experiment."""
    result = analyze_experiment(exp_name)
    if not result:
        return
    
    report = result['report']
    filters = report['filters']
    
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}")
    
    print("\n📋 Configuration:")
    print(f"  Seed: {filters['seed']}")
    print(f"  Quality: {filters['quality']}")
    print(f"  Balance: {filters['balance']}")
    print(f"  Positive subsets: {filters.get('positive_subsets', [])}")
    print(f"  Negative subsets: {filters.get('negative_subsets', [])}")
    
    print("\n📊 File Counts:")
    print(f"  Before balance: {report['counts_before']['pos']} pos / {report['counts_before']['neg']} neg")
    print(f"  Final selection: {report['counts_selected']['pos']} pos / {report['counts_selected']['neg']} neg")
    
    if 'pos_sources' in result:
        print("\n🔍 Positive File Sources:")
        for source, count in sorted(result['pos_sources'].items()):
            pct = (count / result['pos_total']) * 100
            print(f"  {source:60s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n🔍 Negative File Sources:")
        for source, count in sorted(result['neg_sources'].items()):
            pct = (count / result['neg_total']) * 100
            print(f"  {source:60s}: {count:4d} ({pct:5.1f}%)")


def main():
    """Analyze key experiments from stage6 sweep."""
    
    print("Stage 6 Sweep File Selection Analysis")
    print("=" * 80)
    
    # Key experiments to analyze
    experiments = [
        ("stage6_001", "Baseline: balance=T, no subsets"),
        ("stage6_002", "Balance=T, +hardneg_50"),
        ("stage6_010", "Balance=T, +small positives"),
        ("stage6_013", "Balance=T, +small pos + hardneg_50"),
        ("stage6_028", "Balance=F, no subsets (full dataset)"),
        ("stage6_040", "Balance=F, +small pos + hardneg_50 (full)"),
        ("stage6_050", "Balance=F, +small pos + hardneg_50 (seed=789)"),
    ]
    
    for exp_name, description in experiments:
        print(f"\n\n{'#'*80}")
        print(f"# {description}")
        print_experiment_analysis(exp_name)
    
    # Summary statistics
    print("\n\n" + "="*80)
    print("SUMMARY: Subset Representation in Balanced Experiments")
    print("="*80)
    
    # Analyze experiments with balance=True and subsets
    balanced_with_subsets = [
        "stage6_002",  # +hardneg_50
        "stage6_004",  # +hardneg_50 (seed=456)
        "stage6_006",  # +hardneg_50 (seed=789)
        "stage6_013",  # +small + hardneg_50
        "stage6_014",  # +small + hardneg_50 (seed=456)
        "stage6_015",  # +small + hardneg_50 (seed=789)
    ]
    
    print("\nExperiments with balance=True and hardneg_50 subset:")
    print(f"{'Experiment':<15} {'Seed':<6} {'Total Neg':<10} {'From Manifest':<15} {'From hardneg_50':<18} {'Subset %':<10}")
    print("-" * 90)
    
    for exp_name in balanced_with_subsets:
        result = analyze_experiment(exp_name)
        if result and 'neg_sources' in result:
            report = result['report']
            seed = report['filters']['seed']
            manifest_count = result['neg_sources'].get('manifest', 0)
            subset_count = sum(v for k, v in result['neg_sources'].items() if 'hardneg' in k.lower())
            total = result['neg_total']
            pct = (subset_count / total * 100) if total > 0 else 0
            print(f"{exp_name:<15} {seed:<6} {total:<10} {manifest_count:<15} {subset_count:<18} {pct:>6.1f}%")
    
    print("\n✅ Analysis complete! See docs/STAGE6_FILE_SELECTION_ANALYSIS.md for full details.")


if __name__ == "__main__":
    main()
