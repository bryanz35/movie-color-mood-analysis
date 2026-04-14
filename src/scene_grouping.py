"""Group shots into semi-master-shots by LAB color similarity.

Implements the Kim & Choi (ICMR 2020) approach: merge consecutive shots
whose mean CIELAB color is within a threshold (ΔE < 15).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2lab
from tqdm import tqdm


def compute_shot_mean_lab(
    video_path: Path, start_frame: int, end_frame: int, sample_fps: int = 1
) -> np.ndarray:
    """Compute the mean CIELAB color of a shot by sampling frames.

    Args:
        video_path: Path to the video file.
        start_frame: First frame of the shot.
        end_frame: Last frame of the shot.
        sample_fps: Frames per second to sample.

    Returns:
        Mean LAB color as a (3,) ndarray.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps / sample_fps))

    lab_accum = []
    for frame_idx in range(start_frame, end_frame + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        new_w = 320
        new_h = int(h * new_w / w)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        lab = rgb2lab(frame_rgb / 255.0)
        lab_accum.append(lab.reshape(-1, 3).mean(axis=0))

    cap.release()

    if not lab_accum:
        return np.zeros(3)
    return np.mean(lab_accum, axis=0)


def group_shots_into_scenes(
    shots: list[dict], video_path: Path, delta_e_threshold: float = 15.0
) -> list[dict]:
    """Merge consecutive shots into scenes based on LAB color similarity.

    Args:
        shots: List of shot dicts with start_frame/end_frame.
        video_path: Path to the video file.
        delta_e_threshold: Maximum ΔE between consecutive shots to merge.

    Returns:
        List of scene dicts with scene_id, start_frame, end_frame, shot_ids.
    """
    if not shots:
        return []

    mean_labs = []
    for shot in tqdm(shots, desc="Computing shot mean LAB colors"):
        lab = compute_shot_mean_lab(video_path, shot["start_frame"], shot["end_frame"])
        mean_labs.append(lab)

    scenes: list[dict] = []
    current_scene_shots = [0]

    for i in range(1, len(shots)):
        delta_e = float(np.linalg.norm(mean_labs[i] - mean_labs[i - 1]))
        if delta_e < delta_e_threshold:
            current_scene_shots.append(i)
        else:
            scenes.append({
                "scene_id": len(scenes),
                "start_frame": shots[current_scene_shots[0]]["start_frame"],
                "end_frame": shots[current_scene_shots[-1]]["end_frame"],
                "shot_ids": current_scene_shots,
            })
            current_scene_shots = [i]

    # Flush last scene
    scenes.append({
        "scene_id": len(scenes),
        "start_frame": shots[current_scene_shots[0]]["start_frame"],
        "end_frame": shots[current_scene_shots[-1]]["end_frame"],
        "shot_ids": current_scene_shots,
    })

    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Group shots into scenes by LAB similarity")
    parser.add_argument("--shots", type=Path, required=True, help="Shots JSON file")
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, required=True, help="Output scenes JSON")
    parser.add_argument("--threshold", type=float, default=15.0, help="ΔE threshold")
    args = parser.parse_args()

    with open(args.shots) as f:
        shots_data = json.load(f)

    scenes = group_shots_into_scenes(shots_data["shots"], args.video, args.threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"scenes": scenes}, f, indent=2)
    print(f"Grouped {len(shots_data['shots'])} shots into {len(scenes)} scenes → {args.output}")


if __name__ == "__main__":
    main()
