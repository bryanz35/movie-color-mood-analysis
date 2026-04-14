"""Librosa feature extraction per scene (tempo, spectral centroid, RMS, chroma)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def extract_audio_from_video(video_path: Path, output_path: Path) -> None:
    """Demux audio from video using ffmpeg.

    Args:
        video_path: Path to the video file.
        output_path: Path to write the WAV audio.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def extract_scene_audio_features(
    audio: np.ndarray, sr: int, start_frame: int, end_frame: int, video_fps: float
) -> dict:
    """Extract audio features for a scene segment.

    Args:
        audio: Full audio signal as a 1D ndarray.
        sr: Audio sample rate.
        start_frame: Scene start video frame.
        end_frame: Scene end video frame.
        video_fps: Video frame rate.

    Returns:
        Dict with tempo, spectral_centroid_mean, rms_mean, and chroma_mean (12 values).
    """
    start_sample = int(start_frame / video_fps * sr)
    end_sample = int(end_frame / video_fps * sr)
    segment = audio[start_sample:end_sample]

    if len(segment) < sr:  # Less than 1 second
        return {
            "tempo": 0.0,
            "spectral_centroid_mean": 0.0,
            "rms_mean": 0.0,
            "chroma_mean": [0.0] * 12,
        }

    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)

    return {
        "tempo": float(np.asarray(tempo).flat[0]),
        "spectral_centroid_mean": float(spectral_centroid.mean()),
        "rms_mean": float(rms.mean()),
        "chroma_mean": chroma.mean(axis=1).tolist(),
    }


def extract_all_audio_features(
    video_path: Path, scenes: list[dict]
) -> list[dict]:
    """Extract audio features for all scenes.

    Args:
        video_path: Path to the video file.
        scenes: List of scene dicts with scene_id, start_frame, end_frame.

    Returns:
        List of dicts with scene_id and audio features.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        extract_audio_from_video(video_path, tmp_path)
        audio, sr = librosa.load(tmp_path, sr=22050, mono=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    results = []
    for scene in tqdm(scenes, desc="Extracting audio features"):
        features = extract_scene_audio_features(
            audio, sr, scene["start_frame"], scene["end_frame"], video_fps
        )
        results.append({
            "scene_id": scene["scene_id"],
            **features,
        })

    return results
