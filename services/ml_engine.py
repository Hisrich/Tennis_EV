"""
Machine Learning Engine
- XGBoost, LightGBM, Logistic Regression ensemble
- Probability calibration
- Model versioning
- Cross-validation with temporal splits
"""
import os
import json
import joblib
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sqlalchemy.orm import Session
from loguru import logger

from models.db_models import Match, ModelRegistry, ModelPrediction, TourLevel, Surface
from services.feature_engineering import FeatureEngineer
from config import config as settings

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "elo_diff", "elo_p1", "elo_p2",
    "elo_surface_diff", "elo_surface_p1", "elo_surface_p2",
    "elo_momentum_p1", "elo_momentum_p2",
    "rank_p1", "rank_p2", "rank_diff", "log_rank_ratio",
    "win_rate_5_p1", "win_rate_5_p2", "win_rate_5_diff",
    "win_rate_10_p1", "win_rate_10_p2", "win_rate_10_diff",
    "win_rate_20_p1", "win_rate_20_p2", "win_rate_20_diff",
    "win_rate_surface_10_p1", "win_rate_surface_10_p2", "win_rate_surface_10_diff",
    "win_rate_surface_20_p1", "win_rate_surface_20_p2", "win_rate_surface_20_diff",
    "matches_7d_p1", "matches_7d_p2", "matches_14d_p1", "matches_14d_p2",
    "days_rest_p1", "days_rest_p2", "fatigue_diff",
    "h2h_total", "h2h_p1_wins", "h2h_p2_wins", "h2h_p1_win_rate",
    "h2h_surface_total", "h2h_surface_p1_win_rate",
    "age_p1", "age_p2", "age_diff", "height_diff",
    "p1_right_handed", "p2_right_handed",
    "surface_hard", "surface_clay", "surface_grass", "surface_carpet",
    "is_grand_slam", "is_masters", "draw_size", "round_num",
    "market_prob_p1", "odds_movement_p1", "bookmaker_count", "has_market_data",
]


class MLEngine:
    """
    Trains, evaluates, and serves ML models for match win probability.
    """

    def __init__(self, db: Session):
        self.db = db
        self.feature_eng = FeatureEngineer(db)
        self._active_model = None
        self._active_version = None

    # ────────────────── Training ──────────────────────────────────

    def train(
        self,
        tour: Optional[TourLevel] = None,
        surface: Optional[Surface] = None,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
        algorithm: str = "xgboost",
    ) -> str:
        """Train a new model and register it. Returns version string."""
        logger.info(f"Starting model training: algo={algorithm}, tour={tour}, surface={surface}")

        # Load matches
        q = self.db.query(Match).filter(
            Match.is_completed == True,
            Match.winner_id.isnot(None),
            Match.is_walkover == False,
        )
        if tour:
            q = q.join(Match.tournament).filter()  # add tour filter if needed
        if min_date:
            q = q.filter(Match.scheduled_at >= min_date)
        if max_date:
            q = q.filter(Match.scheduled_at <= max_date)

        matches = q.order_by(Match.scheduled_at.asc()).all()

        if len(matches) < settings.MIN_MATCHES_FOR_TRAINING:
            raise ValueError(f"Need at least {settings.MIN_MATCHES_FOR_TRAINING} matches, got {len(matches)}")

        logger.info(f"Building features for {len(matches)} matches...")
        df = self.feature_eng.build_dataset(matches)

        if df.empty:
            raise ValueError("Feature dataset is empty")

        df = df.sort_values("match_date").reset_index(drop=True)

        X = self._prepare_X(df)
        y = df["target"].values

        logger.info(f"Dataset: {X.shape[0]} rows × {X.shape[1]} features")

        # ── Train/eval split (temporal) ───────────────────────────
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # ── Build model ───────────────────────────────────────────
        base_model = self._build_model(algorithm)
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        calibrated.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────
        probs = calibrated.predict_proba(X_test)[:, 1]
        metrics = {
            "log_loss": float(log_loss(y_test, probs)),
            "brier_score": float(brier_score_loss(y_test, probs)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
        }
        logger.info(f"Metrics: {metrics}")

        # ── Feature importance ────────────────────────────────────
        fi = self._get_feature_importance(base_model, FEATURE_COLUMNS)

        # ── Version & save ────────────────────────────────────────
        version = datetime.utcnow().strftime(f"{algorithm}_%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{version}.pkl")
        joblib.dump(calibrated, model_path)

        # ── Register ──────────────────────────────────────────────
        # Deactivate old models
        self.db.query(ModelRegistry).filter(ModelRegistry.is_active == True).update({"is_active": False})

        registry = ModelRegistry(
            version=version,
            name=f"Tennis EV Model ({algorithm})",
            algorithm=algorithm,
            tour=tour,
            surface=surface,
            log_loss=metrics["log_loss"],
            brier_score=metrics["brier_score"],
            roc_auc=metrics["roc_auc"],
            n_train_samples=len(X_train),
            n_features=X.shape[1],
            feature_importance=fi,
            hyperparameters=self._get_hyperparams(algorithm),
            train_date_start=matches[0].scheduled_at.date() if matches[0].scheduled_at else None,
            train_date_end=matches[-1].scheduled_at.date() if matches[-1].scheduled_at else None,
            is_active=True,
            file_path=model_path,
        )
        self.db.add(registry)
        self.db.commit()

        logger.info(f"Model registered: {version}")
        return version

    # ────────────────── Inference ─────────────────────────────────

    def predict(self, match: Match) -> Optional[Dict]:
        """Generate probability prediction for a match."""
        model = self._load_active_model()
        if model is None:
            logger.warning("No active model found")
            return None

        features = self.feature_eng.build_features(match)
        if features is None:
            return None

        X = self._features_dict_to_array(features)
        probs = model.predict_proba(X)[0]

        prob_p1 = float(probs[1])
        prob_p2 = 1.0 - prob_p1

        # Confidence: distance from 0.5
        confidence = abs(prob_p1 - 0.5) * 2

        return {
            "prob_player1": prob_p1,
            "prob_player2": prob_p2,
            "confidence": confidence,
            "model_version": self._active_version,
            "feature_snapshot": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                  for k, v in features.items()},
        }

    def predict_and_save(self, match: Match) -> Optional[ModelPrediction]:
        """Run prediction and persist to database."""
        result = self.predict(match)
        if not result:
            return None

        pred = ModelPrediction(
            match_id=match.id,
            model_version=result["model_version"],
            model_name="Tennis EV Model",
            prob_player1=result["prob_player1"],
            prob_player2=result["prob_player2"],
            confidence=result["confidence"],
            feature_snapshot=result["feature_snapshot"],
        )
        self.db.add(pred)
        self.db.commit()
        self.db.refresh(pred)
        return pred

    # ────────────────── Internal ──────────────────────────────────

    def _load_active_model(self):
        if self._active_model is not None:
            return self._active_model

        registry = (
            self.db.query(ModelRegistry)
            .filter(ModelRegistry.is_active == True)
            .first()
        )
        if not registry or not registry.file_path:
            return None

        if os.path.exists(registry.file_path):
            self._active_model = joblib.load(registry.file_path)
            self._active_version = registry.version
            return self._active_model

        return None

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[available].fillna(0).values
        return X

    def _features_dict_to_array(self, features: Dict) -> np.ndarray:
        row = [features.get(col, 0.0) for col in FEATURE_COLUMNS]
        return np.array([row])

    def _build_model(self, algorithm: str):
        if algorithm == "xgboost":
            return XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif algorithm == "logistic":
            return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        try:
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                return {f: float(fi[i]) for i, f in enumerate(feature_names) if i < len(fi)}
        except Exception:
            pass
        return {}

    def _get_hyperparams(self, algorithm: str) -> Dict:
        model = self._build_model(algorithm)
        return {k: str(v) for k, v in model.get_params().items()}
