# Titanic project/Thien/version9/src/experiment_logger.py
# AI-generated file, do not edit manually.
# This file is part of the Titanic project, version 9.
import os
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------- small utils ----------


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(s: str) -> str:
    return (
        "".join(c.lower() if c.isalnum() else "-" for c in s)
        .strip("-")
        .replace("--", "-")
    )


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_jsonable(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    return obj


def _json_dump(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable, ensure_ascii=False)


def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def hash_features(feature_names: Sequence[str]) -> str:
    arr = ",".join(sorted([str(f) for f in feature_names]))
    return _hash_bytes(arr.encode("utf-8"))


def hash_dataset_sample(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    sample_n: int = 200,
    random_state: int = 42,
) -> str:
    rng = np.random.RandomState(random_state)
    if isinstance(X, pd.DataFrame):
        Xs = X
        if len(X) > sample_n:
            Xs = X.sample(sample_n, random_state=random_state)
        b = pd.util.hash_pandas_object(Xs, index=True).values.tobytes()
    else:
        Xnp = np.array(X)
        if len(Xnp) > sample_n:
            idx = rng.choice(len(Xnp), size=sample_n, replace=False)
            Xnp = Xnp[idx]
        b = Xnp.tobytes()
    if y is not None:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            ys = (
                y
                if len(y) <= sample_n
                else y.sample(sample_n, random_state=random_state)
            )
            b += pd.util.hash_pandas_object(ys, index=True).values.tobytes()
        else:
            ynp = np.array(y)
            if len(ynp) > sample_n:
                idx = rng.choice(len(ynp), size=sample_n, replace=False)
                ynp = ynp[idx]
            b += ynp.tobytes()
    return _hash_bytes(b)


def plot_confusion_matrix_png(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[List[str]],
    save_path: str,
    normalize: bool = True,
):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_labels = labels if labels is not None else [str(i) for i in range(cm.shape[0])]
    plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")
    plt.yticks(range(len(tick_labels)), tick_labels)
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            plt.text(
                j,
                i,
                txt,
                va="center",
                ha="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------- main logger ----------


class ExperimentLogger:
    """
    Minimal experiment logger for Titanic notebook.

    Usage (in your notebook):
        from src.experiment_logger import ExperimentLogger

        base = "/home/.../Titanic project/Thien/version9"
        logger = ExperimentLogger(
            experiments_dir=f"{base}/experiments",
            model_name="Random Forest",
            params={"n_estimators": 200, "max_depth": None},
            notes="baseline features v1"
        )
        logger.start(X_train, y_train, features=X_train.columns)

        # after CV (optional)
        logger.log_cv(scores)  # scores = array-like of fold accuracies

        # after training and validation
        logger.log_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba)  # y_proba optional
        logger.save_model(model)
        logger.save_feature_importance(model, feature_names=list(X_train.columns))

        # finally
        logger.finalize()  # writes metrics.json and appends a row to EXPERIMENTS_LOG.csv
    """

    MASTER_HEADER = [
        "timestamp",
        "run_id",
        "dataset_hash",
        "features_hash",
        "model_name",
        "params_json",
        "cv_folds",
        "cv_mean_acc",
        "cv_std_acc",
        "val_acc",
        "val_f1",
        "val_precision",
        "val_recall",
        "train_time_s",
        "notes",
    ]

    def __init__(
        self,
        experiments_dir: str,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ):
        self.experiments_dir = experiments_dir
        self.model_name = model_name
        self.params = params or {}
        self.notes = notes

        _ensure_dir(self.experiments_dir)
        self.master_csv = os.path.join(self.experiments_dir, "EXPERIMENTS_LOG.csv")

        self.timestamp = _now_ts()
        self.run_id = f"{self.timestamp}__{_slug(self.model_name)}"
        self.run_dir = os.path.join(self.experiments_dir, "runs", self.run_id)
        _ensure_dir(self.run_dir)

        self.metrics: Dict[str, Any] = {}
        self.cv_info: Dict[str, Any] = {}
        self.dataset_hash = ""
        self.features_hash = ""
        self._t0 = None

        # persist params immediately
        _json_dump(self.params, os.path.join(self.run_dir, "params.json"))

    # mark the start and optionally compute hashes
    def start(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        features: Optional[Sequence[str]] = None,
    ):
        self._t0 = time.time()
        if X is not None:
            self.dataset_hash = hash_dataset_sample(X, y)
        if features is not None:
            self.features_hash = hash_features(features)

    def log_cv(self, scores):
        scores = np.array(scores, dtype=float)
        self.cv_info = {
            "cv_folds": int(len(scores)),
            "cv_mean_acc": float(scores.mean()) if len(scores) else None,
            "cv_std_acc": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        }

    def log_metrics(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
    ):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        self.metrics.update(
            {
                "val_acc": float(acc),
                "val_precision": float(prec),
                "val_recall": float(rec),
                "val_f1": float(f1),
            }
        )

        # classification report
        report_txt = classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        )
        with open(
            os.path.join(self.run_dir, "classification_report.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report_txt)

        # confusion matrix plot
        plot_confusion_matrix_png(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            save_path=os.path.join(self.run_dir, "confusion_matrix.png"),
            normalize=True,
        )

        # predictions CSV
        df_pred = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 1:
                df_pred["proba_1"] = y_proba
            elif y_proba.ndim == 2:
                # prefer positive class prob in binary case
                if y_proba.shape[1] == 2:
                    df_pred["proba_1"] = y_proba[:, 1]
                else:
                    for j in range(y_proba.shape[1]):
                        df_pred[f"proba_{j}"] = y_proba[:, j]
        df_pred.to_csv(os.path.join(self.run_dir, "predictions.csv"), index=False)

        # metrics.json
        _json_dump(self.metrics, os.path.join(self.run_dir, "metrics.json"))

    def save_model(self, model: Any, filename: str = "model.joblib"):
        path = os.path.join(self.run_dir, filename)
        joblib.dump(model, path)

    def save_feature_importance(
        self, model: Any, feature_names: Optional[List[str]] = None
    ):
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(len(importances))]
            df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
            df_imp.sort_values("importance", ascending=False, inplace=True)
            df_imp.to_csv(
                os.path.join(self.run_dir, "feature_importance.csv"), index=False
            )

    def save_submission(
        self,
        passenger_ids: Sequence[int],
        predictions: Sequence[int],
        filename: str = "submission.csv",
    ):
        sub = pd.DataFrame(
            {"PassengerId": passenger_ids, "Survived": np.array(predictions, dtype=int)}
        )
        sub.to_csv(os.path.join(self.run_dir, filename), index=False)

    def finalize(self, train_time_s: Optional[float] = None):
        if train_time_s is None and self._t0 is not None:
            train_time_s = float(time.time() - self._t0)
        # aggregate master row
        row = {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "dataset_hash": self.dataset_hash,
            "features_hash": self.features_hash,
            "model_name": self.model_name,
            "params_json": json.dumps(
                self.params, default=_to_jsonable, ensure_ascii=False
            ),
            "cv_folds": self.cv_info.get("cv_folds"),
            "cv_mean_acc": self.cv_info.get("cv_mean_acc"),
            "cv_std_acc": self.cv_info.get("cv_std_acc"),
            "val_acc": self.metrics.get("val_acc"),
            "val_f1": self.metrics.get("val_f1"),
            "val_precision": self.metrics.get("val_precision"),
            "val_recall": self.metrics.get("val_recall"),
            "train_time_s": train_time_s,
            "notes": self.notes,
        }
        # append to master CSV (create with header if missing)
        df_row = pd.DataFrame([row], columns=self.MASTER_HEADER)
        if not os.path.exists(self.master_csv):
            _ensure_dir(os.path.dirname(self.master_csv))
            df_row.to_csv(self.master_csv, index=False)
        else:
            df_row.to_csv(self.master_csv, mode="a", header=False, index=False)
