# Research Notes
---

## Kim & Choi (2020) — Automatic Color Scheme Extraction from Movies
*ICMR '20. Code: https://github.com/SuziKim/ICMR2020-MovieColorSchemer*

Can reimplement most color extraction logic and methodology.

#### Key points:
 - builds a tool for extracting and visualizing color data from films
 - pull colors from each video frame using k-means clustering -> displays in a
   visualization (will need to make a visualization later)
 - paper shows color scheme consistency for Miyazaki's films (validates color
   extraction)

#### Implementation details:
- **shot grouping** — merge adjacent shots if similar frames in `src/scene_grouping.py`.
- **preprocessing conventions** — 24 fps encode, resize to 320px wide. `src/frame_sampling.py`.
- pipeline structure in `scripts/run_pipeline.py`.

---

## Wei, Dimitrova & Chang (2004) — Color-Mood Analysis of Films
*IEEE ICME 2004.*

Methodology behind mood analysis

#### Key points:
 - defines 8 mood categories: passionate, cheerful, humorous, peaceful, gloomy,
   scary, sad, mysterious
 - two representations: Movie Palette Histogram (global) + Mood Dynamics
   Histogram (transitions over time)
 - ~80% SVM accuracy on live-action — sets the bar for our classifier

#### Implementation details:
- **mood taxonomy** — 8 labels for annotation schema in `data/annotations/`.
- **valence-arousal mapping** — passionate/cheerful, peaceful, gloomy/sad, scary/mysterious

---

## Flueckiger & Halter (2020) — Film Colors in Digital Humanities
*DHQ vol. 14 no. 4. ERC FilmColors project, 400+ films analyzed.*

Argument for CIELAB color over RGB/HSV

#### Key points:
 - CIELAB makes more sense for human interpretation (eucliddean distance for
   color)

#### Implementation details:
- **color space** — CIELAB used in `src/color_extraction.py`, makes more sense for mood classification

---

## Isaac (2020) — Web-Based 3D HSV Color Visualization
*3rd International Conference on Web Studies, ACM. Case study: Miyazaki.*

Another color extraction paper for animated films

#### Key points:
 - web tool, JS-based 3D HSV visualization
 - k-means on Miyazaki frames to find palettes (same as above paper)
 - finds measurable palette consistency across a director's filmography

#### Implementation details:
- utilize k-means algorithm, good for films with very intentional color (spiderman)

---

## Hayes (2025) — Disney Color Associations
*Master's thesis, Liberty University.*

Compares mood interpretation for different color schemes

#### Key points:
 - tests whether Disney exposure rewires color-emotion associations (Bandura,
   social learning theory)
 - high-exposure viewers favor Disney conventions (purple/green = villain)
 - low-exposure viewers favor traditional color psychology (dark = villain)
 - heroes trend warm yellows/reds/blues; villains trend black/purple/green

#### Implementation details:
- consider hero color in palettes to affect mood

---

## Halter et al. (2019) — VIAN Annotation Tool
*Computer Graphics Forum vol. 38 no. 3. Open source: https://github.com/FilmColors/VIAN*

VIAN is a desktop app for color analysis. After more review, we won't be using
this and instead will use k-means to segment colors and python to do processing.
Still saving this in case its useful in the future

---

## Souček & Lokoč (2020) — TransNet V2
*arXiv:2008.04838. Pip: `transnetv2-pytorch`.*

ML model for shot boundaries (automated scene segmentation)

#### Key points:
 - 3D conv net with dilated DCNN cells
 - much better than PySceneDetect on gradual transitions (fades, dissolves)
 - outputs (start_frame, end_frame) pairs per shot

#### Implementation details:
- **`src/shot_detection.py`** implements transnetv2
- output feeds into Kim & Choi's semi-master-shot merger to recombine similar
  scenes.

---

## Other notes:
 - Spiderman movie accessed seems to be unfinished cut / not final render. Some
   colors are not as sharp or smooth, but shouldn't affect overall color
   palette. If results are bad, may try with other movies
---
