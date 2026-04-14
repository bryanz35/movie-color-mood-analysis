# CLAUDE.md — Movie Color-Mood Analysis Project

## Project Overview

Quantitative color scheme extraction from movies, segmented by scene mood. The project investigates the relationship between cinematic color palettes and perceived scene mood using computational methods.

Three phases: (1) structural scene segmentation, (2) mood annotation + color extraction, (3) statistical analysis of color-mood associations.

## Tech Stack

- Python 3.11+
- TransNetV2 (`pip install transnetv2-pytorch`) — shot boundary detection
- OpenCV (`opencv-python`) — frame extraction, image processing
- scikit-learn — k-means clustering, classification, statistical tests
- scikit-image — CIELAB color space conversion
- librosa — audio feature extraction (tempo, spectral centroid, RMS, chroma)
- matplotlib / seaborn — visualization
- scipy — Earth Mover's Distance, ANOVA, Kruskal-Wallis
- POT (`pip install POT`) — optimal transport / EMD in high dimensions
- ffmpeg (system) — video/audio demuxing

Optional:
- open_clip — CLIP embeddings for semantic scene grouping
- umap-learn — dimensionality reduction for palette embedding visualization

## Project Structure

```
movie-color-mood/
├── CLAUDE.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── shot_detection.py      # TransNetV2 wrapper, shot boundary extraction
│   ├── scene_grouping.py      # Group shots into semi-master-shots by LAB color similarity
│   ├── frame_sampling.py      # Extract frames at 1 fps, resize to 320px wide
│   ├── color_extraction.py    # K-means in CIELAB, per-scene palette + summary stats
│   ├── audio_features.py      # Librosa feature extraction per scene (tempo, spectral centroid, RMS, chroma)
│   ├── analysis.py            # EMD matrix, ANOVA, classifier, t-SNE/UMAP embedding
│   └── visualization.py       # Movie barcodes, palette strips, polar hue plots, mood-annotated timelines
├── data/
│   ├── videos/                # Input video files (not committed)
│   ├── annotations/           # Mood annotations as JSON: {scene_id, valence, arousal, mood_label}
│   └── outputs/               # Extracted palettes, features, analysis results
├── notebooks/
│   └── exploration.ipynb      # Interactive analysis and figure generation
└── scripts/
    ├── run_pipeline.py        # End-to-end: video → shots → scenes → palettes → analysis
    └── annotate.py            # CLI helper for manual mood annotation
```

## Key Design Decisions

### Scene Segmentation
- Use TransNetV2 for shot boundary detection, NOT PySceneDetect (better accuracy on film content).
- Run TransNetV2 with `--device cpu` for reproducibility. MPS on Apple Silicon has numerical inconsistency (produces different scene counts than CPU).
- Group adjacent shots into "semi-master-shots" (Kim & Choi, ICMR 2020): merge consecutive shots whose mean LAB color is within a threshold (e.g., ΔE < 15). This produces scene-level units.

### Color Extraction
- Convert frames to CIELAB color space (perceptually uniform; Euclidean distance ≈ perceptual color difference ΔE).
- Sample frames at 1 fps. Resize to 320px wide before processing.
- Run k-means (k=5) on pixel colors in LAB space per scene. Weight clusters by pixel count.
- Output per scene: list of (L, a, b, proportion) tuples.
- Summary stats per scene: mean L (luminance), mean chroma (sqrt(a² + b²)), hue histogram (bin atan2(b, a) into 12 bins of 30°).

### Mood Annotation
- Use a valence-arousal model (continuous, 2D) rather than discrete categories for primary annotation.
- Additionally map to 8 mood types from Wei et al.: passionate, cheerful, humorous, peaceful, gloomy, scary, sad, mysterious.
- Store annotations as JSON in `data/annotations/`. Schema:
  ```json
  {
    "film": "film_name",
    "scenes": [
      {
        "scene_id": 0,
        "start_frame": 0,
        "end_frame": 1440,
        "valence": 0.6,
        "arousal": 0.3,
        "mood_label": "peaceful",
        "notes": ""
      }
    ]
  }
  ```
- Keep corpus small: 2–3 films, ~50–100 scenes total.

### Analysis
- Earth Mover's Distance (EMD) between scene palettes in LAB space. Use `scipy.stats.wasserstein_distance` for 1D or `ot.emd2` from POT for full 3D LAB.
- Build pairwise EMD distance matrix across all scenes. Embed with UMAP, color by mood label.
- ANOVA or Kruskal-Wallis on luminance, chroma, dominant hue across mood categories.
- Train random forest classifier: mood label from color features (mean L, mean chroma, hue histogram, top-k palette colors). Report accuracy vs. chance baseline (~12.5% for 8 classes). Target: ~80% (Wei et al. benchmark).
- Audio features (librosa) as validation: check if audio-derived mood correlates with color-derived mood.

## Key References

- Kim & Choi, "Automatic Color Scheme Extraction from Movies" (ICMR 2020). GitHub: https://github.com/SuziKim/ICMR2020-MovieColorSchemer — semi-master-shot concept, saliency-weighted color extraction.
- Wei et al., "Color-Mood Analysis of Films Based on Syntactic and Psychological Models" (ICME 2004) — Movie Palette Histogram, 8 mood types, SVM classification at ~80% accuracy.
- Souček & Lokoč, "TransNet V2" (arXiv 2008.04838) — shot boundary detection neural network.
- ERC FilmColors project (digitalhumanities.org/dhq) — large-scale film color analysis methodology.

## Coding Conventions

- Type hints on all function signatures.
- Docstrings with Args/Returns for public functions.
- No classes unless state management requires it; prefer pure functions.
- Use pathlib.Path for all file paths.
- Print progress with tqdm for any loop over frames or scenes.
- All numerical arrays as numpy ndarrays; pandas DataFrames for tabular results.
- Seed all random operations (k-means, train/test splits) with `random_state=42`.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run shot detection on a video
python -m src.shot_detection --input data/videos/film.mp4 --output data/outputs/film_shots.json

# Group shots into scenes
python -m src.scene_grouping --shots data/outputs/film_shots.json --video data/videos/film.mp4 --output data/outputs/film_scenes.json

# Extract color palettes
python -m src.color_extraction --scenes data/outputs/film_scenes.json --video data/videos/film.mp4 --output data/outputs/film_palettes.json

# Run full pipeline
python scripts/run_pipeline.py --video data/videos/film.mp4 --annotations data/annotations/film.json

# Launch annotation helper
python scripts/annotate.py --scenes data/outputs/film_scenes.json --video data/videos/film.mp4
```

## Known Pitfalls

- TransNetV2 detects *shot* boundaries (cuts), not *scene* boundaries. The scene grouping step is essential.
- Naive k-means on full frames is dominated by background colors. Saliency weighting (as in Kim & Choi) improves results but adds complexity. Start without it; add if palettes look wrong.
- CIELAB conversion requires specifying an illuminant. Use D65 (default in skimage `rgb2lab`).
- Movie barcodes (1px-wide column per frame) are useful for sanity-checking the pipeline visually.
- Opening/closing credits should be trimmed before analysis — they have distinctive palettes unrelated to narrative mood.
