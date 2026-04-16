#!/usr/bin/env python3
"""
train_model.py
==============
Standalone script to pre-train and persist the RandomForest classifier.
Run this once before using main.py if you want to pre-cache the model.

Usage:
    python train_model.py
    python train_model.py --verbose
    python train_model.py --show-importances
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config  # noqa: E402
from ml_model.trainer import ModelTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the repository classifier")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-importances", action="store_true", help="Print top feature importances after training")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of RF trees (default: 200)")
    args = parser.parse_args()

    cfg = Config(verbose=True, model_n_estimators=args.n_estimators)
    print("🌲 Training RandomForest classifier on synthetic data...\n")

    trainer = ModelTrainer(cfg)
    trainer.train_and_save()

    print(f"\n✅ Model saved to: {cfg.model_path}")

    if args.show_importances:
        print("\n📊 Top 15 Feature Importances:")
        print("─" * 50)
        importances = trainer.feature_importances()
        for i, (feature, importance) in enumerate(list(importances.items())[:15], 1):
            bar = "█" * int(importance * 300)
            print(f"  {i:2}. {feature:<35} {bar} {importance:.4f}")


if __name__ == "__main__":
    main()
