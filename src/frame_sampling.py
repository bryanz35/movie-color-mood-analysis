"""Extract frames from video at 1 fps, resized to 320px wide."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def sample_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    sample_fps: int = 1,
    target_width: int = 320,
) -> list[np.ndarray]:
    """Extract and resize frames from a video segment.

    Args:
        video_path: Path to the video file.
        start_frame: First frame index.
        end_frame: Last frame index.
        sample_fps: Frames per second to sample.
        target_width: Width to resize frames to (maintains aspect ratio).

    Returns:
        List of RGB frames as uint8 ndarrays with shape (H, W, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps / sample_fps))

    frames = []
    for frame_idx in range(start_frame, end_frame + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        new_h = int(h * target_width / w)
        frame_rgb = cv2.resize(frame_rgb, (target_width, new_h))
        frames.append(frame_rgb)

    cap.release()
    return frames


def sample_frames_for_scenes(
    video_path: Path, scenes: list[dict], sample_fps: int = 1
) -> dict[int, list[np.ndarray]]:
    """Sample frames for each scene.

    Args:
        video_path: Path to the video file.
        scenes: List of scene dicts with scene_id, start_frame, end_frame.
        sample_fps: Frames per second to sample.

    Returns:
        Dict mapping scene_id to list of RGB frame arrays.
    """
    result = {}
    for scene in tqdm(scenes, desc="Sampling frames"):
        result[scene["scene_id"]] = sample_frames(
            video_path, scene["start_frame"], scene["end_frame"], sample_fps
        )
    return result
