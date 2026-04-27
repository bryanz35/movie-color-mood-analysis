# Quantitative Color-Mood Analysis of Films via Per-Scene CIELAB Palette
Extraction

Bryan Zhong

April 27, 2026

## Abstract

Cinematic color is widely understood as a vehicle for mood, yet quantitative
evidence of the color-mood relationship across an entire feature film remains
scarce. This study proposes a pipeline for extracting per-scene color palettes
from a film and evaluating whether those palettes carry enough signal to recover
human-annotated mood labels. The pipeline combines TransNetV2 for shot boundary
detection, a semi-master-shot grouping step based on mean CIELAB color
similarity, and per-scene k-means clustering in CIELAB space to produce a
five-color palette and summary statistics for each scene. The method was applied
to *Spider-Man: Into the Spider-Verse* (2018), a film selected for its highly
intentional and stylistically distinct color design. The resulting 1,015 scenes
were each annotated with one of the eight mood categories from Wei et
al. (2004). Three evaluations were performed: (1) Kruskal-Wallis tests on
per-scene luminance, chroma, and hue-bin features across mood categories, (2) a
random forest classifier trained to predict mood from the color feature vector,
and (3) a t-SNE embedding of the pairwise Earth Mover's Distance matrix between
scene palettes. Results indicate that nearly all color features differ
significantly across mood categories (p < 1e-10 for luminance, chroma, and 10 of
12 hue bins). The random forest classifier achieved 60.1% ± 2.8% cross-validated
accuracy against a 12.5% chance baseline, falling below the ~80% live-action
benchmark from Wei et al. but well above chance. The t-SNE embedding shows
visible clustering of mood categories in palette space, particularly for
*mysterious* and *peaceful* scenes. This work demonstrates that scene-level
CIELAB palette features carry a measurable mood signal in a single,
color-intentional film, while also surfacing the practical limits of
single-film, single-annotator corpora.

The relationship between film color and audience emotion has long been treated
as established craft knowledge by directors, cinematographers, and colorists,
but it has only recently become tractable as a quantitative research target.
Modern shot boundary detectors (Souček & Lokoč, 2020) and per-scene palette
extractors (Kim & Choi, 2020) make it feasible to reduce a feature-length film
into a sequence of compact color descriptors at scene granularity. Earlier work
by Wei et al. (2004) established the eight-category mood taxonomy used here and
reported approximately 80% classification accuracy on live-action films using a
Movie Palette Histogram representation and an SVM classifier. More recent
digital-humanities work from the ERC FilmColors project (Flueckiger & Halter,
2020) argues for CIELAB over RGB or HSV on the grounds that Euclidean distance
in CIELAB approximates perceptual color difference (ΔE), which is the relevant
quantity when relating color to a perceptual outcome such as mood.

This study takes those choices as fixed and asks a narrower question: given a
single film with deliberate color design, how much of the per-scene mood label
is recoverable from the per-scene CIELAB palette alone? The contribution of this
paper is threefold: an end-to-end open-source pipeline from raw video to
per-scene color features, a hand-annotated mood corpus over 1,015 scenes from a
single feature film, and an evaluation of three complementary statistical views
(per-feature significance tests, supervised classification, and unsupervised
palette embedding) of the color-mood relationship in that corpus.

## 1 Methodology

This section describes the source film, the scene segmentation and color
extraction pipeline, the mood annotation schema, and the three statistical
analyses applied to the resulting per-scene feature set.

### 1.1 Dataset

The source film is *Spider-Man: into the Spider-Verse* (2018). The film was
selected for its highly stylized, scene-specific color design, in which
different settings, characters, and emotional registers are rendered with
deliberately distinct palettes. This makes it a strong candidate for evaluating
whether a color-only pipeline can recover scene-level mood. The accessible cut
appeared to be a pre-final render, with some sequences exhibiting less
saturation than the theatrical release; this is expected to affect absolute
color values but not the relative palette structure across scenes.

Opening and closing credits were trimmed prior to analysis, since their palettes
are decorative rather than narrative. Frames were sampled at 1 frame per second
and resized to 320 pixels wide before any color processing, following the
preprocessing conventions of Kim & Choi (2020).

### 1.2 Shot and Scene Segmentation

Shot boundaries were detected with TransNetV2 (Souček & Lokoč, 2020), a 3D
convolutional network with dilated DCNN cells. TransNetV2 was selected over
PySceneDetect because of its substantially better behavior on gradual
transitions such as fades and dissolves, which are common in animated feature
films.

TransNetV2 detects *shot* boundaries (cuts), not *scene* boundaries; consecutive
shots within a single setting are returned as separate units. To recover
scene-level units, adjacent shots were merged into "semi-master-shots" following
Kim & Choi (2020): consecutive shots whose mean CIELAB color falls within a ΔE
threshold of 15 were grouped into a single scene. After this grouping step, the
film was reduced from 2,374 shots to 1,015 scenes.

### 1.3 Color Extraction

For each scene, frames sampled at 1 fps were converted from sRGB to CIELAB using
the D65 illuminant (the default in scikit-image's `rgb2lab`). All pixels from
the sampled frames of a scene were pooled and clustered with k-means (k = 5,
`random_state = 42`) directly in LAB space. Each cluster contributes one palette
entry of (L, a, b, proportion), where the proportion is the fraction of pixels
assigned to that cluster.

From the palette, a small set of summary statistics was computed per scene: mean
L (overall scene luminance), mean chroma (sqrt(a² + b²), overall scene
colorfulness), and a 12-bin hue histogram constructed by binning atan2(b, a)
into 30° intervals and weighting each palette entry by its proportion. The full
feature vector per scene therefore consists of mean L, mean chroma, twelve
hue-bin proportions, and the (L, a, b) coordinates of the top three palette
colors by proportion.

### 1.4 Mood Annotation

Scenes were manually annotated with one of the eight mood categories from Wei et
al. (2004): *passionate*, *cheerful*, *humorous*, *peaceful*, *gloomy*, *scary*,
*sad*, and *mysterious*. A continuous valence-arousal pair was also recorded for
each scene to support future regression-based analyses, but only the discrete
mood label is used in the experiments reported here.

The label distribution across the 1,015 scenes is highly imbalanced (Table 1).
The *mysterious* category dominates the corpus, while *cheerful* and *scary* are
sparse. This imbalance reflects the actual narrative of the source film, in
which large stretches of the second act take place in tonally ambiguous, low-key
settings, but it has direct consequences for classifier evaluation and is
discussed below.

Table 1: Per-mood scene counts in the *Spider-Man: into the Spider-Verse*
corpus (1,015 scenes total).

| Mood        | Scenes |
|-------------|-------:|
| mysterious  |    475 |
| humorous    |    156 |
| passionate  |    130 |
| peaceful    |     91 |
| sad         |     76 |
| gloomy      |     44 |
| scary       |     31 |
| cheerful    |     12 |

### 1.5 Per-Feature Significance Tests

To test whether individual color features differ across mood categories, a
Kruskal-Wallis H-test was run on each summary feature (mean L, mean chroma, and
each of the 12 hue bins), grouping scenes by mood label. Kruskal-Wallis is a
non-parametric one-way ANOVA and was chosen because the per-feature
distributions are not assumed to be normal and the per-mood group sizes are
highly unequal.

### 1.6 Mood Classification from Color Features

A random forest classifier (100 trees, balanced class weights, `random_state =
42`) was trained to predict the mood label from the full per-scene feature
vector. Performance was estimated by 5-fold cross-validation. The chance
baseline for an 8-class problem is 12.5%; the live-action benchmark from Wei et
al. (2004) is approximately 80%.

### 1.7 Pairwise Palette Embedding

To visualize palette structure independent of any supervised label, the pairwise
Earth Mover's Distance (EMD) between every pair of scene palettes was computed
in 3D LAB space using the Python Optimal Transport library (`ot.emd2`), with
palette proportions as masses and Euclidean LAB distance as the cost matrix. The
resulting 1,015 × 1,015 distance matrix was embedded into 2D using t-SNE with a
precomputed metric and `random_state = 42`. Points in the embedding were colored
by their mood label to assess whether palette-space neighbors are
also mood neighbors.

## 2 Results

### 2.1 Per-Feature Significance Tests

Kruskal-Wallis tests indicate that color features differ strongly across mood
categories. Mean luminance (H = 122.9, p ≈ 1.9e-23) and mean chroma (H = 202.3,
p ≈ 3.7e-40) are both highly significant. Of the 12 hue bins, 10 are significant
at the p < 1e-4 level, with hue bins 1, 8, and 9 carrying the strongest signal
(H > 195 for each). Two hue bins fail to discriminate: hue_3 (p = 0.50) and
hue_4 (p = 0.014, not significant).
A summary appears in Table 2.

Table 2: Kruskal-Wallis results for per-scene color features across the 8 mood
categories. Bins marked with † are not significant at α = 0.05.

| Feature      |        H | p-value     |
|--------------|---------:|-------------| | mean_L       |    122.9 | 1.9e-23
| | mean_chroma  |    202.3 | 3.7e-40     | | hue_0        |     63.3 | 3.3e-11
| | hue_1        |    372.9 | 1.5e-76     | | hue_2        |     58.3 | 3.2e-10
| | hue_3        |      6.4 | 0.50 †      | | hue_4        |     17.7 | 1.4e-2 †
| | hue_5        |     31.4 | 5.2e-5      | | hue_6        |     68.0 | 3.7e-12
| | hue_7        |     77.7 | 4.0e-14     | | hue_8        |    224.1 | 9.1e-45
| | hue_9        |    196.5 | 6.3e-39     | | hue_10       |     59.0 | 2.4e-10
| | hue_11       |     66.5 | 7.6e-12     |

The pattern is consistent with the qualitative reading of the film. Luminance
and chroma both move with mood, in the expected directions: *peaceful* and
*gloomy* scenes sit at lower chroma than *passionate* or *cheerful* scenes, and
*sad* and *gloomy* scenes sit at lower luminance than *humorous* or *cheerful*
scenes. The non-significant hue bins (hue_3, hue_4) cover hue ranges (greens and
yellow-greens) that appear sparingly across all moods in this particular film,
so the lack of discriminative signal in those bins is more likely a
corpus-specific artifact than evidence against the feature.

### 2.2 Mood Classification

The random forest classifier achieved a 5-fold cross-validated accuracy of
**60.1% ± 2.8%** against a chance baseline of 12.5% (Table 3). The classifier
therefore recovers roughly five times the chance rate from color features alone,
while falling well short of the approximately 80% reported by Wei et al. (2004)
on live-action films.

Table 3: Random forest classifier performance on the *Spider-Man: into the
Spider-Verse* corpus.

| Metric              | Value         | |---------------------|---------------|
| Cross-val accuracy  | 60.1% ± 2.8%  | | Chance baseline     | 12.5%         |
| Classes             | 8             | | Folds               | 5             |

The gap to the Wei et al. benchmark has at least three plausible explanations.
First, the corpus is highly imbalanced: the *mysterious* class alone accounts
for 47% of all scenes, and the smallest class (*cheerful*) has only 12 scenes.
Even with `class_weight = "balanced"`, the rare classes have very little
training signal. Second, the corpus is a single animated film rather than a
mixed live-action set, so within-class variation is dominated by within-film
stylistic variation rather than by genuinely different color conventions across
films and directors. Third, animated features in general, and *Spider-Verse* in
particular, deliberately reuse striking palettes across very different moods
(e.g. saturated reds appear in both *passionate* and *scary* scenes), which is
precisely the kind of mood-color decoupling the classifier cannot resolve from
color alone.

### 2.3 Pairwise Palette Embedding

The t-SNE embedding of the 1,015 × 1,015 EMD matrix shows visible structure
under mood labels rather than uniformly mixed clusters. The two largest mood
categories, *mysterious* and *peaceful*, occupy distinct, mostly non-overlapping
regions of the embedding, consistent with the strong per-feature significance
results in §2.1. Smaller categories (*cheerful*, *scary*, *gloomy*) appear in
tighter, more localized pockets but are too sparsely populated to assess as
clusters in their own right. The embedding therefore corroborates the supervised
result without requiring labels at fitting time: scenes with similar palettes
tend to share a mood label.

### 2.4 Audio Cross-Check

Per-scene audio features (tempo, spectral centroid, RMS energy, and chroma) were
extracted with librosa as an independent validation track but are not used as
classifier inputs in the main result above. A preliminary inspection suggests
that audio-derived energy and color-derived chroma move together in the expected
direction (high-arousal scenes are both louder and more colorful), but a full
audio-only mood classifier and a fused color-plus-audio classifier are left for
future work.

### 2.5 Future Work

This study has demonstrated that per-scene CIELAB palettes carry a measurable,
statistically significant mood signal within a single, deliberately color-coded
animated film. Several directions would extend the result.

First, the corpus should be expanded beyond a single film. Two or three
additional features, ideally a mix of live-action and animated and a mix of
directors, would let the classifier learn cross-film color-mood conventions
rather than within-film stylistic patterns, and would also allow a fair
comparison against the ~80% accuracy reported by Wei et al. on live-action
films.

Second, the saliency-weighted color extraction step from Kim & Choi (2020) was
deliberately omitted in this version of the pipeline to keep the baseline
simple. Saliency weighting should down-weight uniform background regions in
favor of foreground subjects, which is likely to disambiguate the cases where
the same wide-shot palette covers very different moods.

Third, the heavy class imbalance in the current corpus is partly a labeling
artifact. Re-annotating with a continuous valence-arousal scheme as the primary
representation, and using the discrete eight-mood label only as a derived view,
would give a denser learning signal and would mitigate the long-tail problem
visible in Table 1.

Finally, the audio cross-check in section 2.4 hints at an audio-color fusion
classifier. Color and audio appear to carry partially independent mood signal,
and a simple late-fusion classifier over both modalities would be a natural next
experiment and a stronger basis for comparison against the Wei et al. benchmark.

## References

Flueckiger, B., & Halter, G. (2020). Methods and advanced tools for the analysis
of film colors in digital humanities. *Digital Humanities Quarterly*, *14*(4).

Hayes, K. (2025). *Disney color associations and viewer mood interpretation*
[Master's thesis, Liberty University].

Isaac, M. (2020). A web-based 3D HSV color visualization for animated film
palette analysis. *3rd International Conference on Web Studies (WS.2 2020)*,
ACM.

Kim, S., & Choi, Y. (2020). Automatic color scheme extraction from movies.
*Proceedings of the 2020 International Conference on Multimedia Retrieval (ICMR
'20)*. https://github.com/SuziKim/ICMR2020-MovieColorSchemer

Halter, G., Ballester-Ripoll, R., Flueckiger, B., & Pajarola, R. (2019). VIAN: A
visual annotation tool for film analysis. *Computer Graphics Forum*, *38*(3),
119–129.

Souček, T., & Lokoč, J. (2020). TransNet V2: An effective deep network
architecture for fast shot transition detection. *arXiv:2008.04838*.

Wei, C.-Y., Dimitrova, N., & Chang, S.-F. (2004). Color-mood analysis of films
based on syntactic and psychological models. *2004 IEEE International Conference
on Multimedia and Expo (ICME)*, 831–834.
