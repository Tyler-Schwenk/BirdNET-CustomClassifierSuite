#!/usr/bin/env python3
"""
Analyze manifest.csv to count positive audio files per recorder per day.

Usage:
    python scripts/analyze_positive_counts_by_date.py
"""

import re
import csv
import statistics
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def extract_date_from_path(path: str) -> str | None:
    """Extract YYYYMMDD date from file path."""
    match = re.search(r'(\d{8})', path)
    if match:
        date_str = match.group(1)
        try:
            # Validate and format as YYYY-MM-DD
            dt = datetime.strptime(date_str, '%Y%m%d')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return None
    return None

def main():
    manifest_path = Path("docs/manifest.csv")
    output_path = Path("positive_counts_by_recorder_date.txt")
    
    print(f"Reading manifest from: {manifest_path}")
    
    # Dictionary to store counts: {(recorder, date): count}
    counts = defaultdict(int)
    skipped = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Only process positive samples
            if row['label'] != 'positive':
                continue
            
            recorder = row['recorder']
            orig_path = row['orig_full_path']
            
            # Extract date from path
            date = extract_date_from_path(orig_path)
            
            if date and recorder:
                counts[(recorder, date)] += 1
            else:
                skipped += 1
    
    # Sort by recorder, then date
    sorted_counts = sorted(counts.items(), key=lambda x: (x[0][0], x[0][1]))
    
    # Write results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("POSITIVE AUDIO FILES COUNT BY RECORDER AND DATE\n")
        f.write("=" * 60 + "\n\n")
        
        current_recorder = None
        recorder_totals = defaultdict(int)
        
        for (recorder, date), count in sorted_counts:
            # Print recorder header when it changes
            if recorder != current_recorder:
                if current_recorder is not None:
                    f.write(f"  {current_recorder} Total: {recorder_totals[current_recorder]} files\n")
                    f.write("-" * 60 + "\n")
                current_recorder = recorder
                f.write(f"\n{recorder}:\n")
            
            f.write(f"  {date}: {count:4d} files\n")
            recorder_totals[recorder] += count
        
        # Print final recorder total
        if current_recorder:
            f.write(f"  {current_recorder} Total: {recorder_totals[current_recorder]} files\n")
            f.write("-" * 60 + "\n")
        
        # Summary statistics
        f.write("\n" + "=" * 60 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total positive files: {sum(counts.values())}\n")
        f.write(f"Total recorders: {len(recorder_totals)}\n")
        f.write(f"Total date-recorder combinations: {len(counts)}\n")
        if skipped > 0:
            f.write(f"Skipped (no date found): {skipped}\n")
        
        # Calculate comprehensive statistics
        daily_counts = list(counts.values())
        mean_count = statistics.mean(daily_counts)
        median_count = statistics.median(daily_counts)
        stdev_count = statistics.stdev(daily_counts) if len(daily_counts) > 1 else 0.0
        min_count = min(daily_counts)
        max_count = max(daily_counts)
        
        # Percentiles
        q25 = statistics.quantiles(daily_counts, n=4)[0]  # 25th percentile
        q75 = statistics.quantiles(daily_counts, n=4)[2]  # 75th percentile
        iqr = q75 - q25
        
        f.write(f"\n--- Central Tendency ---\n")
        f.write(f"Mean (average): {mean_count:.2f} files/day\n")
        f.write(f"Median: {median_count:.1f} files/day\n")
        f.write(f"Standard deviation: {stdev_count:.2f}\n")
        
        f.write(f"\n--- Range & Percentiles ---\n")
        f.write(f"Min: {min_count} files/day\n")
        f.write(f"25th percentile: {q25:.1f} files/day\n")
        f.write(f"75th percentile: {q75:.1f} files/day\n")
        f.write(f"Max: {max_count} files/day\n")
        f.write(f"Interquartile Range (IQR): {iqr:.1f}\n")
        
        # Distribution bins
        bins = {
            "1-10": 0,
            "11-25": 0,
            "26-50": 0,
            "51-100": 0,
            "101-200": 0,
            "201-500": 0,
            "500+": 0
        }
        
        for count in daily_counts:
            if count <= 10:
                bins["1-10"] += 1
            elif count <= 25:
                bins["11-25"] += 1
            elif count <= 50:
                bins["26-50"] += 1
            elif count <= 100:
                bins["51-100"] += 1
            elif count <= 200:
                bins["101-200"] += 1
            elif count <= 500:
                bins["201-500"] += 1
            else:
                bins["500+"] += 1
        
        f.write(f"\n--- Distribution (Days by File Count) ---\n")
        for bin_label, bin_count in bins.items():
            pct = (bin_count / len(daily_counts) * 100) if daily_counts else 0
            f.write(f"{bin_label:>10} files: {bin_count:3d} days ({pct:5.1f}%)\n")
        
        # Typical vs Chorus classification
        # Using 1.5*IQR rule for outliers
        outlier_threshold = q75 + 1.5 * iqr
        typical_nights = [c for c in daily_counts if c <= outlier_threshold]
        chorus_nights = [c for c in daily_counts if c > outlier_threshold]
        
        f.write(f"\n--- Typical vs Chorus Nights ---\n")
        f.write(f"Outlier threshold (Q75 + 1.5*IQR): {outlier_threshold:.1f} files\n")
        f.write(f"Typical nights (<= threshold): {len(typical_nights)} days\n")
        if typical_nights:
            f.write(f"  Mean: {statistics.mean(typical_nights):.1f} files/day\n")
            f.write(f"  Median: {statistics.median(typical_nights):.1f} files/day\n")
        f.write(f"Chorus nights (> threshold): {len(chorus_nights)} days\n")
        if chorus_nights:
            f.write(f"  Mean: {statistics.mean(chorus_nights):.1f} files/day\n")
            f.write(f"  Median: {statistics.median(chorus_nights):.1f} files/day\n")
            f.write(f"  Range: {min(chorus_nights)}-{max(chorus_nights)} files\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("RECORDER TOTALS\n")
        f.write("=" * 60 + "\n")
        for recorder in sorted(recorder_totals.keys()):
            f.write(f"{recorder}: {recorder_totals[recorder]:4d} total positive files\n")
        
        # Per-recorder statistics
        f.write("\n" + "=" * 60 + "\n")
        f.write("PER-RECORDER STATISTICS\n")
        f.write("=" * 60 + "\n")
        
        recorder_day_counts = defaultdict(list)
        for (recorder, date), count in counts.items():
            recorder_day_counts[recorder].append(count)
        
        for recorder in sorted(recorder_day_counts.keys()):
            rec_counts = recorder_day_counts[recorder]
            f.write(f"\n{recorder}:\n")
            f.write(f"  Recording days: {len(rec_counts)}\n")
            f.write(f"  Total files: {sum(rec_counts)}\n")
            f.write(f"  Mean per day: {statistics.mean(rec_counts):.1f}\n")
            f.write(f"  Median per day: {statistics.median(rec_counts):.1f}\n")
            f.write(f"  Range: {min(rec_counts)}-{max(rec_counts)} files/day\n")
        
        # Model confidence implications
        f.write("\n" + "=" * 60 + "\n")
        f.write("MODEL CONFIDENCE IMPLICATIONS\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nFor day-level predictions (assuming file-level metrics):\n")
        f.write(f"\nTypical night scenario (median {median_count:.0f} files):\n")
        f.write(f"  - At 95% precision: expect ~{median_count * 0.05:.1f} false positives/night\n")
        f.write(f"  - At 90% recall: expect to miss ~{median_count * 0.10:.1f} true calls/night\n")
        f.write(f"  - At 90% F1: expect ~{median_count * 0.10:.1f} total errors/night\n")
        
        f.write(f"\nBusy night scenario (75th percentile {q75:.0f} files):\n")
        f.write(f"  - At 95% precision: expect ~{q75 * 0.05:.1f} false positives/night\n")
        f.write(f"  - At 90% recall: expect to miss ~{q75 * 0.10:.1f} true calls/night\n")
        f.write(f"  - At 90% F1: expect ~{q75 * 0.10:.1f} total errors/night\n")
        
        if chorus_nights:
            max_chorus = max(chorus_nights)
            f.write(f"\nChorus night scenario (max {max_chorus} files):\n")
            f.write(f"  - At 95% precision: expect ~{max_chorus * 0.05:.1f} false positives/night\n")
            f.write(f"  - At 90% recall: expect to miss ~{max_chorus * 0.10:.1f} true calls/night\n")
            f.write(f"  - At 90% F1: expect ~{max_chorus * 0.10:.1f} total errors/night\n")
        
        f.write(f"\nNote: Day-level confidence depends on whether you need:\n")
        f.write(f"  - Detection: 'Was species present?' → High confidence even with 1 true positive\n")
        f.write(f"  - Abundance: 'How many calls?' → Confidence scales with file counts\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Results written to: {output_path}")
    print(f"\nKey Statistics:")
    print(f"  - Total positive files: {sum(counts.values())}")
    print(f"  - Recording days: {len(counts)}")
    print(f"  - Median files/day: {statistics.median(daily_counts):.1f}")
    print(f"  - Mean files/day: {statistics.mean(daily_counts):.2f}")
    print(f"  - Typical range (25th-75th): {statistics.quantiles(daily_counts, n=4)[0]:.1f}-{statistics.quantiles(daily_counts, n=4)[2]:.1f}")
    print(f"  - Max files in one day: {max(daily_counts)}")

if __name__ == "__main__":
    main()
