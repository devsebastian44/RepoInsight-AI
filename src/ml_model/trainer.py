"""
ml_model/trainer.py
=====================
Trains a RandomForestClassifier on synthetic data and persists it
alongside a StandardScaler for feature normalization.

Usage:
    from ml_model.trainer import ModelTrainer
    trainer = ModelTrainer(config)
    trainer.train_and_save()     # Train and write to disk
    model, scaler = trainer.load()   # Load from disk
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from ml_model.data_generator import LABEL_MAP, SyntheticDataGenerator


class ModelTrainer:
    """
    Trains the repository-level classifier.

    Pipeline:
        StandardScaler → RandomForestClassifier

    The trained pipeline is pickled to config.model_path.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def train_and_save(self) -> Pipeline:
        """
        Train on synthetic data, evaluate via cross-validation,
        then persist to disk. Returns the fitted pipeline.
        """
        cfg = self.config
        cfg.log("Generating synthetic training data...")

        generator = SyntheticDataGenerator(random_state=cfg.model_random_state)
        X, y = generator.generate()

        cfg.log(f"  Training set: {X.shape[0]} samples × {X.shape[1]} features")
        cfg.log(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        pipeline = self._build_pipeline()

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.model_random_state)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        cfg.log(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Final fit on full data
        pipeline.fit(X, y)

        # Full-data metrics (train set — for sanity check only)
        y_pred = pipeline.predict(X)
        cfg.log("  Classification report (train set):")
        if cfg.verbose:
            labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
            report = classification_report(y, y_pred, target_names=labels)
            for line in report.split("\n"):
                cfg.log(f"    {line}")

        self._save(pipeline)
        return pipeline

    def load(self) -> Pipeline:
        """Load a previously trained pipeline from disk."""
        path = Path(self.config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Run ModelTrainer.train_and_save() first.")
        with open(path, "rb") as f:
            return pickle.load(f)

    def model_exists(self) -> bool:
        return Path(self.config.model_path).exists()

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        """Construct the sklearn Pipeline."""
        clf = RandomForestClassifier(
            n_estimators=self.config.model_n_estimators,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.config.model_random_state,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    def _save(self, pipeline: Pipeline) -> None:
        path = Path(self.config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.config.log(f"  Model saved to {path}")

    def feature_importances(self) -> dict[str, float]:
        """Return a dict of {feature_name: importance} for the trained RF."""
        from dataclasses import fields as dc_fields

        from analysis.feature_engineering import FeatureVector

        pipeline = self.load()
        rf: RandomForestClassifier = pipeline.named_steps["clf"]
        feature_names = [f.name for f in dc_fields(FeatureVector)]
        importances = rf.feature_importances_
        return dict(
            sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        )
