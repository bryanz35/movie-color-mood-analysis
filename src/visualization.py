"""Movie barcodes, palette strips, polar hue plots, and mood-annotated timelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.color import lab2rgb


def lab_to_mpl_color(L: float, a: float, b: float) -> tuple[float, float, float]:
    """Convert a single LAB color to an RGB tuple for matplotlib.

    Args:
        L: Lightness.
        a: Green-red component.
        b: Blue-yellow component.

    Returns:
        (R, G, B) tuple with values in [0, 1].
    """
    lab = np.array([[[L, a, b]]])
    rgb = lab2rgb(lab)[0, 0]
    return tuple(np.clip(rgb, 0, 1))


def plot_palette_strip(palette: list[dict], ax: plt.Axes | None = None) -> plt.Axes:
    """Draw a horizontal palette strip for a single scene.

    Args:
        palette: List of {L, a, b, proportion} dicts.
        ax: Optional matplotlib axes.

    Returns:
        The axes with the palette drawn.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 1))

    x = 0.0
    for color in palette:
        rgb = lab_to_mpl_color(color["L"], color["a"], color["b"])
        width = color["proportion"]
        ax.barh(0, width, left=x, color=rgb, height=1, edgecolor="none")
        x += width

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    return ax


def plot_movie_barcode(
    palettes: list[dict], output_path: Path | None = None
) -> plt.Figure:
    """Create a movie barcode visualization (one column per scene).

    Args:
        palettes: List of scene palette dicts with "palette" key.
        output_path: Optional path to save the figure.

    Returns:
        The matplotlib figure.
    """
    n = len(palettes)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.1), 3))

    for i, scene in enumerate(palettes):
        # Use dominant color (highest proportion)
        dominant = scene["palette"][0]
        rgb = lab_to_mpl_color(dominant["L"], dominant["a"], dominant["b"])
        ax.axvspan(i, i + 1, color=rgb)

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Movie Barcode")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_polar_hue(
    hue_histogram: list[float],
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a polar hue histogram (12 bins of 30 degrees).

    Args:
        hue_histogram: List of 12 values (normalized weights per hue bin).
        title: Plot title.
        ax: Optional polar axes.

    Returns:
        The polar axes.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))

    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    widths = np.full(12, 2 * np.pi / 12)

    bars = ax.bar(angles, hue_histogram, width=widths, bottom=0.0, alpha=0.8)

    # Color each bar by its hue angle
    for bar, angle in zip(bars, angles):
        hue_deg = np.degrees(angle)
        # Approximate hue color: place on a/b circle at chroma=50, L=65
        a = 50 * np.cos(angle)
        b = 50 * np.sin(angle)
        rgb = lab_to_mpl_color(65, a, b)
        bar.set_facecolor(rgb)

    ax.set_title(title, pad=15)
    return ax


def plot_mood_timeline(
    scenes: list[dict],
    annotations: list[dict],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot a mood-annotated timeline with scene colors.

    Args:
        scenes: List of scene palette dicts.
        annotations: List of annotation dicts with mood_label, valence, arousal.
        output_path: Optional path to save the figure.

    Returns:
        The matplotlib figure.
    """
    mood_colors = {
        "passionate": "#e74c3c",
        "cheerful": "#f39c12",
        "humorous": "#2ecc71",
        "peaceful": "#3498db",
        "gloomy": "#7f8c8d",
        "scary": "#2c3e50",
        "sad": "#9b59b6",
        "mysterious": "#1abc9c",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4), sharex=True)

    for i, (scene, ann) in enumerate(zip(scenes, annotations)):
        # Top: scene dominant color
        dominant = scene["palette"][0]
        rgb = lab_to_mpl_color(dominant["L"], dominant["a"], dominant["b"])
        ax1.axvspan(i, i + 1, color=rgb)

        # Bottom: mood label color
        mood = ann.get("mood_label", "")
        mcolor = mood_colors.get(mood, "#cccccc")
        ax2.axvspan(i, i + 1, color=mcolor, alpha=0.7)

    ax1.set_ylabel("Scene Color")
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    ax2.set_ylabel("Mood")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, len(scenes))
    ax2.set_xlabel("Scene Index")

    fig.suptitle("Mood-Annotated Timeline")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_embedding(
    coords: np.ndarray,
    mood_labels: list[str],
    title: str = "Scene Embedding by Mood",
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of 2D scene embeddings colored by mood.

    Args:
        coords: (n, 2) array of embedding coordinates.
        mood_labels: Mood label per scene.
        title: Plot title.
        output_path: Optional path to save the figure.

    Returns:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_moods = sorted(set(mood_labels))
    palette = sns.color_palette("husl", len(unique_moods))
    mood_to_color = dict(zip(unique_moods, palette))

    for mood in unique_moods:
        mask = [m == mood for m in mood_labels]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=mood, color=mood_to_color[mood], s=60, alpha=0.8,
        )

    ax.legend(title="Mood")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
