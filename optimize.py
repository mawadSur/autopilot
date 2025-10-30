from __future__ import annotations

import shutil
from pathlib import Path

import optuna

from train_model import TrainConfig, build_parser, train


def build_config(trial: optuna.Trial) -> TrainConfig:
    parser = build_parser()
    args = parser.parse_args([])

    # Hyperparameters suggested by Optuna
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    args.dropout = trial.suggest_float("dropout", 0.05, 0.5)
    args.num_layers = trial.suggest_int("num_layers", 1, 4)
    args.hidden_size = trial.suggest_int("hidden_size", 64, 512)

    # Speed up trials
    args.epochs = 5
    args.accumulate = 1
    args.calibrate_temp = False
    args.use_class_weights = False
    args.use_focal_loss = False
    args.amp = False
    args.workers = 0

    output_dir = Path("optuna_runs") / f"trial_{trial.number}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir = str(output_dir)
    meta_path = Path(args.meta_path)
    if not meta_path.is_absolute():
        meta_path = output_dir / meta_path
    args.meta_path = str(meta_path)

    cfg = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        meta_path=args.meta_path,
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        accumulate=args.accumulate,
        seed=args.seed,
        price_col=args.price_col,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        time_limit=args.time_limit,
        amp=args.amp,
        workers=args.workers,
        chunksize=args.chunksize,
        task=args.task,
        horizon=args.horizon,
        model_type=args.model_type,
        max_folds=2,
        use_class_weights=args.use_class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        calibrate_temp=args.calibrate_temp,
    )
    return cfg


def objective(trial: optuna.Trial) -> float:
    cfg = build_config(trial)
    score = train(cfg)
    return float(score)


def main() -> None:
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50)

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
