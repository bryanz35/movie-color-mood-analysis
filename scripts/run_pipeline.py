"""End-to-end pipeline: video → shots → scenes → palettes → analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.shot_detection import detect_shots, save_shots
from src.scene_grouping import group_shots_into_scenes
from src.color_extraction import extract_scene_palettes
from src.audio_features import extract_all_audio_features
from src.auto_annotate import generate_annotations
from src.analysis import build_emd_matrix, build_feature_matrix, run_anova, train_mood_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full movie color-mood pipeline")
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Mood annotations JSON (auto-generated if omitted)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--delta-e", type=float, default=15.0, help="ΔE threshold for scene grouping")
    parser.add_argument("--n-colors", type=int, default=5, help="Palette size")
    args = parser.parse_args()

    film_name = args.video.stem
    output_dir = args.output_dir or Path("data/outputs") / film_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Shot detection
    print("\n=== Step 1: Shot Detection ===")
    shots = detect_shots(args.video)
    shots_path = output_dir / "shots.json"
    save_shots(shots, shots_path)
    print(f"  {len(shots)} shots detected")

    # Step 2: Scene grouping
    print("\n=== Step 2: Scene Grouping ===")
    scenes = group_shots_into_scenes(shots, args.video, args.delta_e)
    scenes_path = output_dir / "scenes.json"
    with open(scenes_path, "w") as f:
        json.dump({"scenes": scenes}, f, indent=2)
    print(f"  {len(scenes)} scenes formed")

    # Step 3: Color extraction
    print("\n=== Step 3: Color Extraction ===")
    palettes = extract_scene_palettes(args.video, scenes, args.n_colors)
    palettes_path = output_dir / "palettes.json"
    with open(palettes_path, "w") as f:
        json.dump({"palettes": palettes}, f, indent=2)
    print(f"  Palettes extracted for {len(palettes)} scenes")

    # Step 4: Audio features
    print("\n=== Step 4: Audio Features ===")
    audio_features = extract_all_audio_features(args.video, scenes)
    audio_path = output_dir / "audio_features.json"
    with open(audio_path, "w") as f:
        json.dump({"audio_features": audio_features}, f, indent=2)
    print(f"  Audio features for {len(audio_features)} scenes")

    # Step 5: Annotations (auto-generate if not supplied)
    print("\n=== Step 5: Mood Annotations ===")
    if args.annotations is not None and args.annotations.exists():
        print(f"  Using supplied annotations: {args.annotations}")
        with open(args.annotations) as f:
            annotations_data = json.load(f)
    else:
        annotations_data = generate_annotations(
            film_name, scenes, palettes, audio_features
        )
        annotations_path = (
            args.annotations
            if args.annotations is not None
            else Path("data/annotations") / f"{film_name}.json"
        )
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(annotations_path, "w") as f:
            json.dump(annotations_data, f, indent=2)
        print(f"  Auto-generated {len(annotations_data['scenes'])} annotations → {annotations_path}")

    mood_labels = [s["mood_label"] for s in annotations_data["scenes"]]

    # Step 6: Analysis
    print("\n=== Step 6: Analysis ===")

    # EMD matrix
    emd_matrix = build_emd_matrix(palettes)
    emd_path = output_dir / "emd_matrix.npy"
    import numpy as np
    np.save(emd_path, emd_matrix)

    # Feature matrix + ANOVA
    features = build_feature_matrix(palettes)
    anova_results = run_anova(features, mood_labels)
    anova_path = output_dir / "anova_results.json"
    with open(anova_path, "w") as f:
        json.dump(anova_results, f, indent=2)

    # Classifier
    clf_results = train_mood_classifier(features, mood_labels)
    clf_path = output_dir / "classifier_results.json"
    with open(clf_path, "w") as f:
        json.dump(clf_results, f, indent=2)
    print(
        f"  Classifier accuracy: {clf_results['mean_accuracy']:.2%} "
        f"± {clf_results.get('std_accuracy', 0.0):.2%} "
        f"(chance: {clf_results['chance_baseline']:.2%}, "
        f"majority: {clf_results.get('majority_baseline', 0.0):.2%})"
    )
    if "macro_f1" in clf_results:
        print(
            f"  Macro-F1: {clf_results['macro_f1']:.3f}  "
            f"Weighted-F1: {clf_results['weighted_f1']:.3f}  "
            f"Dummy-stratified macro-F1: {clf_results['dummy_stratified']['macro_f1']:.3f}"
        )
        print("  Per-class F1:")
        for cls, m in clf_results["per_class"].items():
            print(
                f"    {cls:12s} P={m['precision']:.2f}  R={m['recall']:.2f}  "
                f"F1={m['f1']:.2f}  n={m['support']}"
            )
        print("  Top features:")
        for entry in clf_results["feature_importances"][:5]:
            print(f"    {entry['feature']:18s} {entry['importance']:.3f}")

    print(f"\n=== Done. Results in {output_dir} ===")


if __name__ == "__main__":
    main()
