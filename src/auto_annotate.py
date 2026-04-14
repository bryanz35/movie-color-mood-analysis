"""Automatic mood annotation from color + audio features.

Maps per-scene color summary stats and audio features to a point in
valence-arousal space, then assigns the nearest of 8 mood labels
(Wei et al., ICME 2004).

Valence proxy: warm-hue proportion + normalized luminance (warm/bright
colors correlate with positive affect).
Arousal proxy: chroma + tempo + RMS energy (saturated colors, fast
tempo, loud audio correlate with high arousal).

Audio features are z-scored within the film to be film-relative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MOOD_CENTROIDS: dict[str, tuple[float, float]] = {
    "passionate": (0.6, 0.8),
    "cheerful":   (0.9, 0.4),
    "humorous":   (0.7, 0.0),
    "peaceful":   (0.5, -0.5),
    "sad":        (-0.5, -0.5),
    "gloomy":     (-0.7, -0.2),
    "mysterious": (-0.3, 0.3),
    "scary":      (-0.6, 0.7),
}


def _warm_score(hue_histogram: list[float]) -> float:
    """Signed warm-vs-cool score in [-1, 1] from a 12-bin hue histogram.

    Bins are 30° each starting at 0°. Warm: 0-90° and 330-360°
    (bins 0, 1, 2, 11). Cool: 150-270° (bins 5, 6, 7, 8).
    """
    h = np.asarray(hue_histogram)
    if h.sum() <= 0:
        return 0.0
    warm = float(h[[0, 1, 2, 11]].sum())
    cool = float(h[[5, 6, 7, 8]].sum())
    return warm - cool


def _tanh_z(values: np.ndarray) -> np.ndarray:
    """Z-score then squash via tanh to [-1, 1]."""
    mu = values.mean()
    sigma = values.std()
    if sigma < 1e-9:
        return np.zeros_like(values)
    return np.tanh((values - mu) / sigma)


def compute_valence_arousal(
    palettes: list[dict], audio_features: list[dict]
) -> list[tuple[float, float]]:
    """Compute per-scene (valence, arousal) in [-1, 1]^2.

    Args:
        palettes: Per-scene palette dicts from extract_scene_palettes.
        audio_features: Per-scene audio feature dicts from
            extract_all_audio_features. Must align by scene_id.

    Returns:
        List of (valence, arousal) tuples, one per scene.
    """
    audio_by_id = {a["scene_id"]: a for a in audio_features}

    warm = np.array([_warm_score(p["summary_stats"]["hue_histogram"]) for p in palettes])
    L = np.array([p["summary_stats"]["mean_L"] for p in palettes])
    chroma = np.array([p["summary_stats"]["mean_chroma"] for p in palettes])

    tempo = np.array([audio_by_id[p["scene_id"]]["tempo"] for p in palettes])
    rms = np.array([audio_by_id[p["scene_id"]]["rms_mean"] for p in palettes])

    L_norm = np.clip((L - 50.0) / 50.0, -1.0, 1.0)
    chroma_norm = np.clip(chroma / 50.0, 0.0, 1.0)

    tempo_z = _tanh_z(tempo)
    rms_z = _tanh_z(rms)

    valence = 0.6 * warm + 0.4 * L_norm
    arousal = 0.4 * chroma_norm + 0.3 * tempo_z + 0.3 * rms_z

    valence = np.clip(valence, -1.0, 1.0)
    arousal = np.clip(arousal, -1.0, 1.0)

    return [(float(v), float(a)) for v, a in zip(valence, arousal)]


def va_to_mood(valence: float, arousal: float) -> str:
    """Map a (valence, arousal) point to the nearest of 8 mood labels."""
    best_label = None
    best_dist = float("inf")
    for label, (cv, ca) in MOOD_CENTROIDS.items():
        d = (valence - cv) ** 2 + (arousal - ca) ** 2
        if d < best_dist:
            best_dist = d
            best_label = label
    assert best_label is not None
    return best_label


def generate_annotations(
    film_name: str,
    scenes: list[dict],
    palettes: list[dict],
    audio_features: list[dict],
) -> dict:
    """Build an annotations dict matching the manual annotation schema.

    Args:
        film_name: Film identifier (used as "film" key).
        scenes: Scene dicts (scene_id, start_frame, end_frame).
        palettes: Per-scene palette dicts.
        audio_features: Per-scene audio feature dicts.

    Returns:
        Dict ready to serialize as JSON to data/annotations/<film>.json.
    """
    va = compute_valence_arousal(palettes, audio_features)
    scenes_by_id = {s["scene_id"]: s for s in scenes}

    annotated = []
    for palette, (v, a) in zip(palettes, va):
        scene = scenes_by_id[palette["scene_id"]]
        annotated.append({
            "scene_id": scene["scene_id"],
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "valence": round(v, 4),
            "arousal": round(a, 4),
            "mood_label": va_to_mood(v, a),
            "notes": "auto-generated",
        })

    return {"film": film_name, "scenes": annotated}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mood annotations from palettes + audio features"
    )
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--palettes", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--film", type=str, default=None)
    args = parser.parse_args()

    with open(args.scenes) as f:
        scenes = json.load(f)["scenes"]
    with open(args.palettes) as f:
        palettes = json.load(f)["palettes"]
    with open(args.audio) as f:
        audio = json.load(f)["audio_features"]

    film_name = args.film or args.output.stem
    annotations = generate_annotations(film_name, scenes, palettes, audio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Wrote {len(annotations['scenes'])} auto-annotations → {args.output}")


if __name__ == "__main__":
    main()
