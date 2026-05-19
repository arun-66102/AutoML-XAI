from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    optuna = None

try:
    import shap
except Exception:
    shap = None

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    SMOTE = None
    ImbPipeline = None


@dataclass
class PreparedData:
    x: pd.DataFrame
    y: pd.Series | None
    target: str | None
    numeric: list[str]
    categorical: list[str]


def run_automl(df: pd.DataFrame, understanding: dict[str, Any], feedback: str = "", model_path: Path | None = None) -> dict[str, Any]:
    task_type = _normalize_task(understanding.get("task_type", "regression"))
    target = _choose_target(df, understanding, task_type)
    prepared = _prepare_dataframe(df, target, task_type)
    feedback = feedback.lower()
    tree_only = "tree" in feedback or "xgboost" in feedback

    if task_type == "regression":
        result = _run_supervised(prepared, _regression_models(tree_only), "regression")
    elif task_type == "classification":
        result = _run_supervised(prepared, _classification_models(tree_only, feedback), "classification")
    elif task_type == "clustering":
        result = _run_clustering(prepared)
    else:
        result = _run_anomaly_detection(prepared, feedback)

    result["task_type"] = task_type
    result["target_column"] = target
    result["features_used"] = prepared.x.columns.tolist()
    result["feature_importance"] = _feature_importance(result.get("best_estimator"), prepared.x, prepared.y)
    result["xai_summary"] = _xai_summary(result, prepared)
    result["recommendations"] = _recommendations(result, understanding, feedback)

    if result.get("best_estimator") is not None:
        if model_path:
            joblib.dump(result["best_estimator"], model_path)
            result["model_file"] = str(model_path)
        result.pop("best_estimator", None)

    return result


def save_model_path(job_id: str, model_dir: Path) -> Path:
    return model_dir / f"{job_id}_best_model.pkl"


def _normalize_task(task: str) -> str:
    task = (task or "").lower().replace(" ", "_")
    if task in {"anomaly", "outlier", "anomaly_detection"}:
        return "anomaly_detection"
    if task in {"classification", "regression", "clustering"}:
        return task
    return "regression"


def _choose_target(df: pd.DataFrame, understanding: dict[str, Any], task_type: str) -> str | None:
    if task_type in {"anomaly_detection", "clustering"}:
        return None
    requested = understanding.get("target_column")
    if requested in df.columns:
        return requested
    candidates = [
        col for col in df.columns
        if any(token in col.lower() for token in ["target", "label", "failure", "fault", "status", "energy", "load", "power"])
    ]
    if candidates:
        return candidates[0]
    return df.columns[-1]


def _prepare_dataframe(df: pd.DataFrame, target: str | None, task_type: str) -> PreparedData:
    data = df.copy()
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[f"{col}_hour"] = data[col].dt.hour
            data[f"{col}_dayofweek"] = data[col].dt.dayofweek
            data = data.drop(columns=[col])
        elif data[col].dtype == "object":
            parsed = pd.to_datetime(data[col], errors="coerce")
            if parsed.notna().mean() > 0.75:
                data[f"{col}_hour"] = parsed.dt.hour
                data[f"{col}_dayofweek"] = parsed.dt.dayofweek
                data = data.drop(columns=[col])

    if target and target in data.columns:
        y = data[target]
        x = data.drop(columns=[target])
        if task_type == "classification" and y.dtype == "object":
            y = y.astype("category").cat.codes
    else:
        y = None
        x = data

    x = x.replace([np.inf, -np.inf], np.nan)
    nunique = x.nunique(dropna=True)
    x = x.loc[:, nunique > 1]
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = x.select_dtypes(exclude="number").columns.tolist()
    return PreparedData(x=x, y=y, target=target, numeric=numeric, categorical=categorical)


def _preprocessor(prepared: PreparedData) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, prepared.numeric), ("cat", categorical_pipe, prepared.categorical)],
        remainder="drop",
    )


def _regression_models(tree_only: bool) -> dict[str, Any]:
    models = {}
    if not tree_only:
        models["Linear Regression"] = LinearRegression()
        models["SVR"] = SVR(C=5.0, epsilon=0.1)
    models["Random Forest Regressor"] = RandomForestRegressor(n_estimators=160, random_state=42)
    if XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(n_estimators=180, max_depth=4, learning_rate=0.07, random_state=42)
    return models


def _classification_models(tree_only: bool, feedback: str) -> dict[str, Any]:
    class_weight = "balanced" if "false negative" in feedback or "imbalance" in feedback else None
    models = {}
    if not tree_only:
        models["Logistic Regression"] = LogisticRegression(max_iter=800, class_weight=class_weight)
        models["SVM"] = SVC(probability=True, class_weight=class_weight)
    models["Random Forest Classifier"] = RandomForestClassifier(n_estimators=180, random_state=42, class_weight=class_weight)
    models["Gradient Boosting Classifier"] = GradientBoostingClassifier(random_state=42)
    if XGBClassifier is not None:
        models["XGBoost Classifier"] = XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.07, eval_metric="logloss", random_state=42)
    return models


def _run_supervised(prepared: PreparedData, models: dict[str, Any], mode: str) -> dict[str, Any]:
    if prepared.y is None:
        raise ValueError("A target column is required for supervised learning.")

    stratify = prepared.y if mode == "classification" and prepared.y.nunique() > 1 and prepared.y.value_counts().min() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        prepared.x, prepared.y, test_size=0.25, random_state=42, stratify=stratify
    )
    leaderboard = []
    best = {"score": -math.inf, "name": None, "pipe": None, "metrics": {}}

    for name, model in models.items():
        pipe = _model_pipeline(prepared, model, mode, y_train)
        tuned_pipe = _maybe_tune(pipe, x_train, y_train, mode, name)
        tuned_pipe.fit(x_train, y_train)
        preds = tuned_pipe.predict(x_test)
        metrics = _regression_metrics(y_test, preds) if mode == "regression" else _classification_metrics(tuned_pipe, x_test, y_test, preds)
        score = metrics["r2"] if mode == "regression" else metrics["f1"]
        leaderboard.append({"model": name, "metrics": metrics})
        if score > best["score"]:
            best = {"score": score, "name": name, "pipe": tuned_pipe, "metrics": metrics}

    return {
        "best_model": best["name"],
        "best_metrics": best["metrics"],
        "leaderboard": sorted(leaderboard, key=lambda row: row["metrics"].get("r2", row["metrics"].get("f1", 0)), reverse=True),
        "best_estimator": best["pipe"],
        "prediction_preview": _preview_predictions(best["pipe"], prepared.x.head(20)),
    }


def _model_pipeline(prepared: PreparedData, model: Any, mode: str, y_train: pd.Series) -> Pipeline:
    steps = [("preprocess", _preprocessor(prepared))]
    if mode == "classification" and SMOTE is not None and ImbPipeline is not None:
        counts = y_train.value_counts()
        if len(counts) > 1 and counts.min() >= 3 and (counts.min() / counts.max()) < 0.45:
            steps.append(("smote", SMOTE(random_state=42, k_neighbors=min(5, int(counts.min()) - 1))))
            steps.append(("model", model))
            return ImbPipeline(steps)
    steps.append(("model", model))
    return Pipeline(steps)


def _maybe_tune(pipe: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, mode: str, name: str) -> Pipeline:
    if optuna is None or len(x_train) < 50 or "Random Forest" not in name:
        return pipe

    def objective(trial):
        params = {
            "model__n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "model__max_depth": trial.suggest_int("max_depth", 3, 12),
            "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        }
        pipe.set_params(**params)
        x_sub, x_val, y_sub, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=trial.number)
        pipe.fit(x_sub, y_sub)
        preds = pipe.predict(x_val)
        return r2_score(y_val, preds) if mode == "regression" else f1_score(y_val, preds, average="weighted", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8, show_progress_bar=False)
    pipe.set_params(**{f"model__{k}": v for k, v in study.best_params.items()})
    return pipe


def _regression_metrics(y_true, preds) -> dict[str, float]:
    return {
        "rmse": round(float(mean_squared_error(y_true, preds, squared=False)), 4),
        "mae": round(float(mean_absolute_error(y_true, preds)), 4),
        "r2": round(float(r2_score(y_true, preds)), 4),
    }


def _classification_metrics(pipe: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, preds) -> dict[str, float]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
    }
    try:
        if hasattr(pipe, "predict_proba") and y_test.nunique() == 2:
            proba = pipe.predict_proba(x_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
    except Exception:
        pass
    return metrics


def _run_anomaly_detection(prepared: PreparedData, feedback: str) -> dict[str, Any]:
    sensitivity = 0.08 if "sensitivity" in feedback or "more anomalies" in feedback else 0.04
    models = {
        "Isolation Forest": IsolationForest(contamination=sensitivity, random_state=42),
        "DBSCAN": DBSCAN(eps=1.8 if sensitivity < 0.06 else 1.3, min_samples=5),
    }
    x_processed = _preprocessor(prepared).fit_transform(prepared.x)
    leaderboard = []
    best_name, best_estimator, best_score, best_labels = None, None, -math.inf, None
    for name, model in models.items():
        labels = model.fit_predict(x_processed)
        anomaly_rate = float(np.mean(labels == -1))
        score = abs(sensitivity - anomaly_rate) * -1
        leaderboard.append({"model": name, "metrics": {"anomaly_rate": round(anomaly_rate, 4), "sensitivity_target": sensitivity}})
        if score > best_score:
            best_name, best_estimator, best_score, best_labels = name, model, score, labels
    return {
        "best_model": best_name,
        "best_metrics": {"anomaly_rate": round(float(np.mean(best_labels == -1)), 4), "anomaly_count": int(np.sum(best_labels == -1))},
        "leaderboard": leaderboard,
        "best_estimator": Pipeline([("preprocess", _preprocessor(prepared)), ("model", best_estimator)]).fit(prepared.x),
        "prediction_preview": [{"row": int(i), "status": "anomaly" if label == -1 else "normal"} for i, label in enumerate(best_labels[:40])],
    }


def _run_clustering(prepared: PreparedData) -> dict[str, Any]:
    x_processed = _preprocessor(prepared).fit_transform(prepared.x)
    leaderboard = []
    best = {"score": -math.inf, "k": 3, "labels": None}
    for k in range(2, min(7, len(prepared.x) - 1)):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(x_processed)
        score = silhouette_score(x_processed, labels) if len(set(labels)) > 1 else -1
        leaderboard.append({"model": f"KMeans k={k}", "metrics": {"silhouette": round(float(score), 4)}})
        if score > best["score"]:
            best = {"score": score, "k": k, "labels": labels}
    estimator = Pipeline([("preprocess", _preprocessor(prepared)), ("model", KMeans(n_clusters=best["k"], random_state=42, n_init="auto"))]).fit(prepared.x)
    return {
        "best_model": f"KMeans k={best['k']}",
        "best_metrics": {"silhouette": round(float(best["score"]), 4), "clusters": int(best["k"])},
        "leaderboard": leaderboard,
        "best_estimator": estimator,
        "prediction_preview": [{"row": int(i), "cluster": int(label)} for i, label in enumerate(best["labels"][:40])],
    }


def _preview_predictions(pipe: Pipeline, rows: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        preds = pipe.predict(rows)
        return [{"row": int(i), "prediction": float(v) if isinstance(v, (np.floating, float, int, np.integer)) else str(v)} for i, v in enumerate(preds)]
    except Exception:
        return []


def _feature_importance(estimator: Any, x: pd.DataFrame, y: pd.Series | None) -> list[dict[str, Any]]:
    if estimator is None:
        return []
    try:
        model = estimator.named_steps.get("model")
        preprocess = estimator.named_steps.get("preprocess")
        names = preprocess.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.ravel(np.abs(model.coef_))
        else:
            return _variance_importance(x)
        rows = sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:12]
        return [{"feature": str(name).replace("num__", "").replace("cat__", ""), "importance": round(float(value), 5)} for name, value in rows]
    except Exception:
        return _variance_importance(x)


def _variance_importance(x: pd.DataFrame) -> list[dict[str, Any]]:
    numeric = x.select_dtypes(include="number")
    if numeric.empty:
        return []
    values = numeric.var(numeric_only=True).sort_values(ascending=False).head(12)
    return [{"feature": str(k), "importance": round(float(v), 5)} for k, v in values.items()]


def _xai_summary(result: dict[str, Any], prepared: PreparedData) -> str:
    top = result.get("feature_importance", [])[:3]
    if not top:
        return "The model used available industrial signals after automatic cleaning and encoding. Add richer sensor tags for deeper explainability."
    names = ", ".join(item["feature"] for item in top)
    if result.get("task_type") == "anomaly_detection":
        return f"Anomaly decisions are most influenced by operating patterns represented by {names}. Review these signals around abnormal windows."
    return f"The selected model relies most on {names}. These variables should be monitored as leading industrial drivers for the requested outcome."


def _recommendations(result: dict[str, Any], understanding: dict[str, Any], feedback: str) -> list[str]:
    context = understanding.get("industrial_context", "industrial analytics")
    task = result.get("task_type")
    recs = [
        f"Deploy the {result.get('best_model')} pipeline as a baseline for {context}.",
        "Validate the selected model on a recent holdout period before connecting it to operator workflows.",
    ]
    if task == "classification":
        recs.append("Tune alert thresholds with maintenance teams to balance missed failures and nuisance alarms.")
    elif task == "regression":
        recs.append("Track residual drift against production data to catch sensor calibration or process changes.")
    elif task == "anomaly_detection":
        recs.append("Route high-confidence anomalies to a review queue with the top contributing sensor tags.")
    else:
        recs.append("Use clusters as operating modes and compare energy, alarm, and downtime behavior across modes.")
    if feedback:
        recs.append(f"Feedback applied: {feedback[:120]}.")
    return recs
