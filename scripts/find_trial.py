import json
from pathlib import Path

# Path to your autotune trials
AUTOTUNE_DIR = Path(r"D:\important\projects\Frog\scripts\BirdNET_FrogTraining\autotune\birdnet_analyzer")

best_trial = None
best_score = -1.0

for trial_dir in AUTOTUNE_DIR.glob("trial_*"):
    trial_file = trial_dir / "trial.json"
    if not trial_file.exists():
        continue
    with open(trial_file, "r") as f:
        trial_data = json.load(f)

    score = None
    # Try keras-tuner style val_AUPRC
    metrics = trial_data.get("metrics", {})
    if "val_AUPRC" in metrics:
        if isinstance(metrics["val_AUPRC"], dict):
            score = metrics["val_AUPRC"].get("best", None)
        else:
            score = metrics["val_AUPRC"]
    # Try BirdNET’s "score" field (negative AUPRC)
    if score is None and "score" in trial_data:
        score = -float(trial_data["score"])  # flip sign back to positive AUPRC

    if score is None:
        continue

    if score > best_score:
        best_score = score
        best_trial = trial_data

if not best_trial:
    print("No completed trials found.")
else:
    print(f"Best val_AUPRC: {best_score:.4f}")
    params = best_trial["hyperparameters"]["values"]

    # Flatten learning_rate_N
    lr = None
    for k, v in params.items():
        if k.startswith("learning_rate_"):
            lr = v

    print("\nYAML-ready training_args:")
    print("training:")
    print(f"  batch_size: {params['batch_size']}")
    print("training_args:")
    print(f"  hidden_units: {params['hidden_units']}")
    print(f"  dropout: {params['dropout']}")
    if lr is not None:
        print(f"  learning_rate: {lr}")
    print(f"  upsampling_ratio: {params.get('upsampling_ratio', 0.0)}")
    if params.get("upsampling_ratio", 0.0) > 0:
        print(f"  upsampling_mode: {params.get('upsampling_mode', 'repeat')}")
    print(f"  mixup: {params.get('mixup', False)}")
    print(f"  label_smoothing: {params.get('label_smoothing', False)}")
    print(f"  focal-loss: {params.get('focal_loss', False)}")
    if params.get("focal_loss", False):
        print(f"  focal-loss-gamma: {params.get('focal_loss_gamma')}")
        print(f"  focal-loss-alpha: {params.get('focal_loss_alpha')}")
