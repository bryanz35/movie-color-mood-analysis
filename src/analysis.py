"""EMD matrix, ANOVA, classifier, and t-SNE/UMAP embedding for color-mood analysis."""

from __future__ import annotations

import numpy as np
import ot
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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

    Args:
        features: Feature DataFrame from build_feature_matrix.
        mood_labels: List of mood label strings.
        random_state: Random seed.

    Returns:
        Dict with mean_accuracy, std_accuracy, chance_baseline, n_classes.
    """
    feature_cols = [c for c in features.columns if c != "scene_id"]
    X = features[feature_cols].values
    y = np.array(mood_labels)

    n_classes = len(set(y))
    chance = 1.0 / n_classes

    clf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight="balanced"
    )

    n_splits = min(5, min(np.bincount(pd.factorize(y)[0])))
    if n_splits < 2:
        return {
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "chance_baseline": chance,
            "n_classes": n_classes,
            "note": "Not enough samples per class for cross-validation",
        }

    scores = cross_val_score(clf, X, y, cv=n_splits, scoring="accuracy")

    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "chance_baseline": chance,
        "n_classes": n_classes,
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
            n_components=2, metric="precomputed", random_state=42, perplexity=min(30, len(emd_matrix) - 1)
        ).fit_transform(emd_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    return embedding
