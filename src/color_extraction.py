"""K-means color extraction in CIELAB space, per-scene palette and summary stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.frame_sampling import sample_frames


def extract_palette(
    frames: list[np.ndarray], n_colors: int = 5, random_state: int = 42
) -> list[dict]:
    """Extract a color palette from a set of frames using k-means in LAB space.

    Args:
        frames: List of RGB uint8 frames.
        n_colors: Number of palette colors to extract.
        random_state: Random seed for k-means.

    Returns:
        List of dicts with keys: L, a, b, proportion (sorted by proportion desc).
    """
    if not frames:
        return []

    all_pixels = []
    for frame in frames:
        lab = rgb2lab(frame / 255.0)
        all_pixels.append(lab.reshape(-1, 3))

    pixels = np.vstack(all_pixels)

    # Subsample if too many pixels for efficiency
    max_pixels = 100_000
    if len(pixels) > max_pixels:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[indices]

    kmeans = KMeans(n_clusters=n_colors, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)

    counts = np.bincount(labels, minlength=n_colors)
    proportions = counts / counts.sum()

    palette = []
    for i in range(n_colors):
        center = kmeans.cluster_centers_[i]
        palette.append({
            "L": float(center[0]),
            "a": float(center[1]),
            "b": float(center[2]),
            "proportion": float(proportions[i]),
        })

    palette.sort(key=lambda c: c["proportion"], reverse=True)
    return palette


def compute_summary_stats(palette: list[dict]) -> dict:
    """Compute summary color statistics from a palette.

    Args:
        palette: List of palette color dicts with L, a, b, proportion.

    Returns:
        Dict with mean_L, mean_chroma, and hue_histogram (12 bins of 30 degrees).
    """
    if not palette:
        return {"mean_L": 0.0, "mean_chroma": 0.0, "hue_histogram": [0.0] * 12}

    weights = np.array([c["proportion"] for c in palette])
    L_vals = np.array([c["L"] for c in palette])
    a_vals = np.array([c["a"] for c in palette])
    b_vals = np.array([c["b"] for c in palette])

    mean_L = float(np.average(L_vals, weights=weights))
    chroma = np.sqrt(a_vals**2 + b_vals**2)
    mean_chroma = float(np.average(chroma, weights=weights))

    # Hue histogram: 12 bins of 30 degrees
    hue_angles = np.degrees(np.arctan2(b_vals, a_vals)) % 360
    hue_histogram = np.zeros(12)
    for angle, weight in zip(hue_angles, weights):
        bin_idx = int(angle // 30) % 12
        hue_histogram[bin_idx] += weight

    # Normalize
    total = hue_histogram.sum()
    if total > 0:
        hue_histogram = hue_histogram / total

    return {
        "mean_L": mean_L,
        "mean_chroma": mean_chroma,
        "hue_histogram": hue_histogram.tolist(),
    }


def extract_scene_palettes(
    video_path: Path, scenes: list[dict], n_colors: int = 5
) -> list[dict]:
    """Extract color palettes for all scenes.

    Args:
        video_path: Path to the video file.
        scenes: List of scene dicts with scene_id, start_frame, end_frame.
        n_colors: Number of palette colors per scene.

    Returns:
        List of dicts with scene_id, palette, and summary_stats.
    """
    results = []
    for scene in tqdm(scenes, desc="Extracting palettes"):
        frames = sample_frames(video_path, scene["start_frame"], scene["end_frame"])
        palette = extract_palette(frames, n_colors=n_colors)
        stats = compute_summary_stats(palette)
        results.append({
            "scene_id": scene["scene_id"],
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "palette": palette,
            "summary_stats": stats,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract color palettes per scene")
    parser.add_argument("--scenes", type=Path, required=True, help="Scenes JSON file")
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, required=True, help="Output palettes JSON")
    parser.add_argument("--n-colors", type=int, default=5, help="Palette size")
    args = parser.parse_args()

    with open(args.scenes) as f:
        scenes_data = json.load(f)

    palettes = extract_scene_palettes(args.video, scenes_data["scenes"], args.n_colors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"palettes": palettes}, f, indent=2)
    print(f"Extracted palettes for {len(palettes)} scenes → {args.output}")


if __name__ == "__main__":
    main()
