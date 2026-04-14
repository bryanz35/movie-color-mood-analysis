"""CLI helper for manual mood annotation of scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MOOD_LABELS = [
    "passionate", "cheerful", "humorous", "peaceful",
    "gloomy", "scary", "sad", "mysterious",
]


def show_scene_preview(video_path: Path, start_frame: int, end_frame: int) -> None:
    """Display a grid of sampled frames from a scene.

    Args:
        video_path: Path to the video file.
        start_frame: First frame of the scene.
        end_frame: Last frame of the scene.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps))  # ~1 fps

    frames = []
    for idx in range(start_frame, end_frame + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            new_w = 320
            new_h = int(h * new_w / w)
            frame = cv2.resize(frame, (new_w, new_h))
            frames.append(frame)
        if len(frames) >= 6:
            break
    cap.release()

    if not frames:
        print("  (no frames to show)")
        return

    # Pad to 6 frames
    while len(frames) < 6:
        frames.append(np.zeros_like(frames[0]))

    # Arrange as 2 rows x 3 cols
    row1 = np.hstack(frames[:3])
    row2 = np.hstack(frames[3:6])
    grid = np.vstack([row1, row2])

    cv2.imshow("Scene Preview", grid)
    cv2.waitKey(1)


def annotate_scenes(scenes_path: Path, video_path: Path, output_path: Path) -> None:
    """Interactive annotation loop.

    Args:
        scenes_path: Path to scenes JSON.
        video_path: Path to the video file.
        output_path: Path to write annotations JSON.
    """
    with open(scenes_path) as f:
        scenes_data = json.load(f)

    scenes = scenes_data["scenes"]
    film_name = video_path.stem

    # Load existing annotations if resuming
    annotations: list[dict] = []
    start_idx = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        annotations = existing.get("scenes", [])
        start_idx = len(annotations)
        print(f"Resuming from scene {start_idx}/{len(scenes)}")

    print(f"\nAnnotating {len(scenes)} scenes from '{film_name}'")
    print(f"Mood labels: {', '.join(f'{i}={m}' for i, m in enumerate(MOOD_LABELS))}")
    print("Enter valence (0-1), arousal (0-1), mood number. Type 'q' to quit.\n")

    for i in range(start_idx, len(scenes)):
        scene = scenes[i]
        print(f"--- Scene {i}/{len(scenes) - 1} "
              f"(frames {scene['start_frame']}–{scene['end_frame']}) ---")

        show_scene_preview(video_path, scene["start_frame"], scene["end_frame"])

        try:
            raw = input("  valence arousal mood# [notes]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving and exiting.")
            break

        if raw.lower() == "q":
            print("Saving and exiting.")
            break

        parts = raw.split()
        if len(parts) < 3:
            print("  Skipping (need at least 3 values)")
            continue

        valence = float(parts[0])
        arousal = float(parts[1])
        mood_idx = int(parts[2])
        notes = " ".join(parts[3:]) if len(parts) > 3 else ""

        annotations.append({
            "scene_id": scene["scene_id"],
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "valence": np.clip(valence, 0.0, 1.0).item(),
            "arousal": np.clip(arousal, 0.0, 1.0).item(),
            "mood_label": MOOD_LABELS[mood_idx % len(MOOD_LABELS)],
            "notes": notes,
        })

        # Save after each annotation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"film": film_name, "scenes": annotations}, f, indent=2)

    cv2.destroyAllWindows()
    print(f"\nSaved {len(annotations)} annotations → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate scenes with mood labels")
    parser.add_argument("--scenes", type=Path, required=True, help="Scenes JSON file")
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, default=None, help="Output annotations JSON")
    args = parser.parse_args()

    output = args.output or Path("data/annotations") / f"{args.video.stem}.json"
    annotate_scenes(args.scenes, args.video, output)


if __name__ == "__main__":
    main()
