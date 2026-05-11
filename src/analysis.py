"""EMD matrix, ANOVA, classifier, and t-SNE/UMAP embedding for color-mood analysis."""

from __future__ import annotations

import numpy as np
import ot
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from tqdm import tqdm


def build_emd_matrix(palettes: list[dict]) -> np.ndarray:
    """Build a pairwise Earth Mover's Distance matrix between scene palettes.

    Uses POT (ot.emd2) for full 3D LAB distance.

    Args:
        palettes: List of scene palette dicts, each with a "palette" key
                  containing list of {L, a, b, proportion} dicts.

    Returns:
        Symmetric (n, n) distance matrix as ndarray.
    """
    n = len(palettes)
    matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Building EMD matrix"):
        for j in range(i + 1, n):
            p_i = palettes[i]["palette"]
            p_j = palettes[j]["palette"]

            # Positions in LAB space
            locs_i = np.array([[c["L"], c["a"], c["b"]] for c in p_i])
            locs_j = np.array([[c["L"], c["a"], c["b"]] for c in p_j])

            # Weights (proportions)
            w_i = np.array([c["proportion"] for c in p_i])
            w_j = np.array([c["proportion"] for c in p_j])

            # Normalize weights
            w_i = w_i / w_i.sum()
            w_j = w_j / w_j.sum()

            # Cost matrix: pairwise Euclidean distance in LAB
            cost = np.linalg.norm(locs_i[:, None] - locs_j[None, :], axis=2)

            emd = ot.emd2(w_i, w_j, cost)
            matrix[i, j] = emd
            matrix[j, i] = emd

    return matrix


def build_feature_matrix(palettes: list[dict]) -> pd.DataFrame:
    """Build a feature matrix from palette summary statistics.

    Args:
        palettes: List of scene palette dicts with summary_stats.

    Returns:
        DataFrame with columns: scene_id, mean_L, mean_chroma, hue_0..hue_11,
        and top palette colors (L, a, b for top 3).
    """
    rows = []
    for p in palettes:
        s = p["summary_stats"]
        row = {
            "scene_id": p["scene_id"],
            "mean_L": s["mean_L"],
            "mean_chroma": s["mean_chroma"],
        }
        for i, h in enumerate(s["hue_histogram"]):
            row[f"hue_{i}"] = h

        # Top 3 palette colors
        for i, c in enumerate(p["palette"][:3]):
            row[f"color_{i}_L"] = c["L"]
            row[f"color_{i}_a"] = c["a"]
            row[f"color_{i}_b"] = c["b"]

        rows.append(row)

    return pd.DataFrame(rows)


def run_anova(features: pd.DataFrame, mood_labels: list[str]) -> dict:
    """Run ANOVA / Kruskal-Wallis tests on color features across mood categories.

    Args:
        features: Feature DataFrame from build_feature_matrix.
        mood_labels: List of mood label strings aligned with features.

    Returns:
        Dict mapping feature name to {statistic, p_value, test} for each test.
    """
    features = features.copy()
    features["mood"] = mood_labels

    test_cols = ["mean_L", "mean_chroma"] + [f"hue_{i}" for i in range(12)]
    results = {}

    for col in test_cols:
        groups = [group[col].values for _, group in features.groupby("mood")]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            continue

        # Use Kruskal-Wallis (non-parametric, no normality assumption)
        stat, p_value = stats.kruskal(*groups)
        results[col] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "test": "kruskal_wallis",
        }

    return results


def train_mood_classifier(
    features: pd.DataFrame, mood_labels: list[str], random_state: int = 42
) -> dict:
    """Train a random forest classifier for mood from color features.

    Reports cross-validated accuracy, macro/weighted F1, per-class
    precision/recall/F1, confusion matrix, feature importances, and two
    dummy-classifier baselines (uniform chance + majority class).

    Args:
        features: Feature DataFrame from build_feature_matrix.
        mood_labels: List of mood label strings.
        random_state: Random seed.

    Returns:
        Dict with cross-validated metrics, per-class report, confusion
        matrix, feature importances, and baselines.
    """
    feature_cols = [c for c in features.columns if c != "scene_id"]
    X = features[feature_cols].values
    y = np.array(mood_labels)

    classes = sorted(set(y))
    n_classes = len(classes)
    uniform_chance = 1.0 / n_classes
    class_counts = {c: int((y == c).sum()) for c in classes}
    majority_baseline = max(class_counts.values()) / len(y)

    clf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight="balanced"
    )

    min_class = min(class_counts.values())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return {
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "chance_baseline": uniform_chance,
            "majority_baseline": majority_baseline,
            "n_classes": n_classes,
            "class_counts": class_counts,
            "note": "Not enough samples per class for cross-validation",
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Accuracy across folds (mean ± std).
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1_macro_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    f1_weighted_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")

    # Out-of-fold predictions for confusion matrix + per-class report.
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    report = classification_report(
        y, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    # Trim to per-class entries + averages, cast floats.
    per_class = {
        c: {
            "precision": float(report[c]["precision"]),
            "recall": float(report[c]["recall"]),
            "f1": float(report[c]["f1-score"]),
            "support": int(report[c]["support"]),
        }
        for c in classes
    }

    cm = confusion_matrix(y, y_pred, labels=classes)

    # Dummy baselines for honest comparison under imbalance.
    dummy_majority = DummyClassifier(strategy="most_frequent")
    dummy_stratified = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy_majority_acc = cross_val_score(dummy_majority, X, y, cv=cv, scoring="accuracy").mean()
    dummy_majority_f1m = cross_val_score(dummy_majority, X, y, cv=cv, scoring="f1_macro").mean()
    dummy_stratified_acc = cross_val_score(dummy_stratified, X, y, cv=cv, scoring="accuracy").mean()
    dummy_stratified_f1m = cross_val_score(dummy_stratified, X, y, cv=cv, scoring="f1_macro").mean()

    # Feature importances from a single fit on all data (informational).
    clf_full = RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight="balanced"
    ).fit(X, y)
    importances = sorted(
        zip(feature_cols, clf_full.feature_importances_.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )

    return {
        "mean_accuracy": float(acc_scores.mean()),
        "std_accuracy": float(acc_scores.std()),
        "macro_f1": float(f1_macro_scores.mean()),
        "macro_f1_std": float(f1_macro_scores.std()),
        "weighted_f1": float(f1_weighted_scores.mean()),
        "weighted_f1_std": float(f1_weighted_scores.std()),
        "chance_baseline": uniform_chance,
        "majority_baseline": majority_baseline,
        "dummy_majority": {
            "accuracy": float(dummy_majority_acc),
            "macro_f1": float(dummy_majority_f1m),
        },
        "dummy_stratified": {
            "accuracy": float(dummy_stratified_acc),
            "macro_f1": float(dummy_stratified_f1m),
        },
        "n_classes": n_classes,
        "n_splits": int(n_splits),
        "class_counts": class_counts,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist(),
        },
        "feature_importances": [
            {"feature": f, "importance": float(v)} for f, v in importances
        ],
    }


def embed_scenes(emd_matrix: np.ndarray, method: str = "umap") -> np.ndarray:
    """Embed scenes in 2D from the EMD distance matrix.

    Args:
        emd_matrix: Pairwise EMD distance matrix.
        method: Embedding method, "umap" or "tsne".

    Returns:
        (n, 2) ndarray of 2D coordinates.
    """
    if method == "umap":
        from umap import UMAP
        embedding = UMAP(
            n_components=2, metric="precomputed", random_state=42
        ).fit_transform(emd_matrix)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        embedding = TSNE(
            n_components=2, metric="precomputed", init="random", random_state=42, perplexity=min(30, len(emd_matrix) - 1)
        ).fit_transform(emd_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    return embedding
