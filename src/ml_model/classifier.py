"""
ml_model/classifier.py
========================
High-level interface for repository-level classification.

Loads (or auto-trains) the persisted RandomForest pipeline and
produces a ClassificationResult with label, confidence, and scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from analysis.feature_engineering import FeatureVector
from ml_model.data_generator import LABEL_MAP


# ── Result Model ──────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Output of the ML classification step.

    Attributes:
        level          : Predicted developer level label (e.g. "Senior")
        level_index    : 0 = Junior, 1 = Mid-level, 2 = Senior
        confidence     : Probability for the predicted class (0.0–1.0)
        probabilities  : Probability per class {Junior, Mid-level, Senior}
        composite_score: Blended score 0–100 combining ML + quality metrics
    """
    level: str
    level_index: int
    confidence: float
    probabilities: dict[str, float]
    composite_score: float
    feature_importances: dict[str, float]


# ── Classifier ────────────────────────────────────────────────────────────────

class RepositoryClassifier:
    """
    Wraps the sklearn pipeline for single-sample prediction.

    On first use, auto-trains the model if no saved model is found.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._pipeline = None

    def predict(self, features: FeatureVector) -> ClassificationResult:
        """
        Predict the repository level from a FeatureVector.

        Returns:
            ClassificationResult with label, confidence, and composite score.
        """
        pipeline = self._load_or_train()
        X = features.to_numpy().reshape(1, -1)

        pred_idx   = int(pipeline.predict(X)[0])
        pred_proba = pipeline.predict_proba(X)[0]

        label      = LABEL_MAP[pred_idx]
        confidence = float(pred_proba[pred_idx])
        proba_dict = {LABEL_MAP[i]: round(float(p), 3) for i, p in enumerate(pred_proba)}

        composite = self._composite_score(pred_idx, confidence, features)

        # Feature importances from RF
        rf = pipeline.named_steps["clf"]
        from dataclasses import fields as dc_fields
        fn = [f.name for f in dc_fields(FeatureVector)]
        importances = dict(sorted(
            zip(fn, rf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        ))
        top_importances = dict(list(importances.items())[:10])

        return ClassificationResult(
            level=label,
            level_index=pred_idx,
            confidence=round(confidence, 3),
            probabilities=proba_dict,
            composite_score=composite,
            feature_importances=top_importances,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_or_train(self):
        """Return the cached pipeline, loading or training as needed."""
        if self._pipeline is not None:
            return self._pipeline

        from ml_model.trainer import ModelTrainer
        trainer = ModelTrainer(self.config)

        if trainer.model_exists():
            self.config.log("Loading persisted ML model...")
            self._pipeline = trainer.load()
        else:
            self.config.log("No saved model found — training now...")
            self._pipeline = trainer.train_and_save()

        return self._pipeline

    def _composite_score(
        self,
        pred_idx: int,
        confidence: float,
        fv: FeatureVector,
    ) -> float:
        """
        Compute a 0–100 composite score blending:
          - ML level prediction  (40 %)
          - Code quality score   (35 %)
          - Dev practices        (15 %)
          - Activity signals     (10 %)
        """
        # ML component: Junior=25, Mid=55, Senior=85 (weighted by confidence)
        level_baseline = [25.0, 55.0, 85.0][pred_idx]
        ml_score = level_baseline * confidence + level_baseline * (1 - confidence) * 0.8

        # Quality component (already 0–100)
        quality = fv.quality_score

        # Practice component
        practice = min(100.0, (
            fv.has_ci * 20 +
            fv.has_tests * 25 +
            fv.has_docker * 10 +
            fv.has_requirements * 10 +
            (fv.best_practice_count / 12) * 35
        ))

        # Activity component
        activity = min(100.0, (
            min(fv.commits_log / 8.0, 1.0) * 40 +
            min(fv.commit_frequency * 2, 1.0) * 30 +
            fv.commit_message_quality * 30
        ))

        composite = (
            ml_score * 0.40 +
            quality  * 0.35 +
            practice * 0.15 +
            activity * 0.10
        )
        return round(min(100.0, max(0.0, composite)), 1)
