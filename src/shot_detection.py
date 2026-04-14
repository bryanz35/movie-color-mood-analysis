"""TransNetV2 wrapper for shot boundary detection.

Detects shot (cut) boundaries in a video file using TransNetV2.
Outputs a list of shot boundaries as frame indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def detect_shots(video_path: Path, device: str = "cpu") -> list[dict]:
    """Run TransNetV2 on a video and return shot boundaries.

    Args:
        video_path: Path to the input video file.
        device: Device to run inference on. Use "cpu" for reproducibility.

    Returns:
        List of dicts with keys: shot_id, start_frame, end_frame.
    """
    from transnetv2_pytorch import TransNetV2

    import torch

    model = TransNetV2()
    model.eval()
    video_frames, single_frame_preds, all_frame_preds = model.predict_video(
        str(video_path)
    )

    if isinstance(single_frame_preds, torch.Tensor):
        single_frame_preds = single_frame_preds.detach().cpu().numpy()

    scenes = model.predictions_to_scenes(single_frame_preds)

    shots = []
    for i, (start, end) in enumerate(tqdm(scenes, desc="Processing shots")):
        shots.append({
            "shot_id": i,
            "start_frame": int(start),
            "end_frame": int(end),
        })

    return shots


def save_shots(shots: list[dict], output_path: Path) -> None:
    """Save shot boundaries to a JSON file.

    Args:
        shots: List of shot boundary dicts.
        output_path: Path to write the JSON output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"shots": shots}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect shot boundaries with TransNetV2")
    parser.add_argument("--input", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    shots = detect_shots(args.input, device=args.device)
    save_shots(shots, args.output)
    print(f"Detected {len(shots)} shots → {args.output}")


if __name__ == "__main__":
    main()
