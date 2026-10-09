"""Experiment runner per EXP-SPEC.md: train on HAM10000, evaluate on DDI, save metrics.

Usage:
    python run_experiment.py --exp baseline
    python run_experiment.py --exp augmented
    python run_experiment.py --exp all
"""
import logging
from argparse import ArgumentParser

from src.evaluation import evaluate_experiment
from src.fairness import compute_and_save_fairness
from src.training import train_model


def run(experiment: str, epochs=None) -> None:
    augment = experiment == "augmented"
    train_kwargs = {"epochs": epochs} if epochs else {}
    train_model(experiment, augment=augment, **train_kwargs)
    result = evaluate_experiment(experiment)
    fairness = compute_and_save_fairness(experiment)
    m = result["overall"]
    gap = fairness["fairness_gap"]
    gap_str = f"{gap:.3f}" if gap is not None else "n/a"
    print(f"[{experiment}] DDI accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} "
          f"recall={m['recall']:.3f} f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f} "
          f"fairness_gap(Light-Dark)={gap_str}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = ArgumentParser(description="Train and evaluate melanoma classifier experiments.")
    parser.add_argument("--exp", choices=["baseline", "augmented", "all"], required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs (default: config.EPOCHS).")
    args = parser.parse_args()

    experiments = ["baseline", "augmented"] if args.exp == "all" else [args.exp]
    for exp in experiments:
        run(exp, args.epochs)
