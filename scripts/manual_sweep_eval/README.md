
# BirdNET Sweep Evaluation Toolkit

This is a slim, modular toolkit for reviewing sweep CSVs across stages.
- Aggregates by config (seed-agnostic), reports mean ± std across seeds.
- Ranks configs by a chosen metric (default: `ood.best_f1.f1`).
- Minimal assumptions: metric columns are discovered via split/metric tokens.
- No magic numbers; behavior is tuned via `constants.py`.

## Quickstart

```bash
python -m birdnet_sweep_toolkit.review --csv stage4sweep.csv --metric ood.best_f1.f1 --top-k 25 --out-csv ranked.csv
```
