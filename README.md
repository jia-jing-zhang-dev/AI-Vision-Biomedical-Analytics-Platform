# AI Vision & Biomedical Analytics Platform

[![tests](https://github.com/jia-jing-zhang-dev/AI-Vision-Biomedical-Analytics-Platform/actions/workflows/tests.yml/badge.svg)](https://github.com/jia-jing-zhang-dev/AI-Vision-Biomedical-Analytics-Platform/actions/workflows/tests.yml)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://visionserve-full-stack-ml-framework.streamlit.app/)

A deep-learning web app for microscopy images. Its main feature analyzes **blood smears**: it finds each white blood cell in a photo, classifies it into one of eight cell types, and flags the cells it is unsure about for a person to check; for single cells it can also show which parts of the image the model looked at. The app also segments cell nuclei and includes classic computer-vision and machine-learning modules.

**Live demo:** <https://visionserve-full-stack-ml-framework.streamlit.app/> (the free host sleeps when idle; the first load can take a minute)

## Why this project

Counting and classifying white blood cells (a *differential count*) is a routine lab test, done by experts at a microscope. The models built here first score 96–99% on the public **BloodMNIST** benchmark, but such scores are measured on images from the same hospital and scanner the model was trained on. The question this project focuses on is whether a model still works on **images from another lab**.

It does not, by default. On a second, independently labelled dataset from a different laboratory and scanner (KU-Optofil PBC, 4,949 cells from 55 patients, never used for training), models that score 96–99% on BloodMNIST fell to 65–76%. Two causes were found and fixed:

1. **Framing:** the other scanner shows each cell at about half the size, so cells that look small in the frame are re-framed to the scale the model was trained on.
2. **Model choice and colour:** instead of training from scratch, the app uses **DinoBloom-S**, a foundation model pretrained on 380,000 blood-cell images, with a small classifier trained here on the training images plus colour-varied copies of them.

The deployed model reaches **95.8%** on the other lab's images and keeps **93.9%** under strong simulated colour changes, where the custom CNN drops to 12%. Every setting was chosen on validation data, and each test set was evaluated once.

## How it works

```
Photo of a blood smear
  → find stained (purple) cells: blue minus green channel, threshold, merge nucleus lobes
  → crop each cell the way the training images are framed (one cell, centred)
  → classify each crop at 224×224 px with DinoBloom-S + a linear classifier
  → confidence below 50%? → mark "needs review" instead of counting it
  → differential count (count per cell type) + per-cell gallery
```

A single centred cell is classified directly (re-framed first if it is small in the frame, i.e. a zoomed-out image), with an optional Grad-CAM heatmap of where the model looked.

## Key results

| What was measured | Result |
|---|---|
| **Blood cells from another lab and scanner** (KU-Optofil PBC, 4,949 cells, 55 patients, never trained on) | DinoBloom-S: **95.8%** (macro-F1 0.79; 93.9% with strong simulated colour shifts). ResNet-18 at 224 px: 76.2%; custom CNN: 65.0% |
| Blood cells, same source (BloodMNIST, 8 classes) | DinoBloom-S 98.9%, ResNet-18 at 224 px 99.1% (not a significant difference); custom CNN 96.7% (fully connected baseline: 72.9%) |
| Finding cells in smears (simulated smears built from held-out images) | 98% of nucleated cells found; 99.0% of them classified correctly |
| **Nucleus segmentation** on expert-annotated microscopy (BBBC039 test set) | U-Net trained here: F1 **0.951** at IoU 0.5, on par with the pretrained Cellpose model (0.938) |
| Touching-cell segmentation, synthetic images (50 images) | Instance F1 **0.80** watershed vs. 0.60 Otsu; Cellpose reference 0.90 |
| FashionMNIST clothing classification (10 classes) | CNN **92.8%** (fully connected baseline: 86.9%) |

Details, methods and limitations are in the sections below: [Higher Resolution, Foundation Models and a Second Scanner](#higher-resolution-foundation-models-and-a-second-scanner-blood-cells), [Blood-smear uploads](#blood-smear-uploads), [Real-Data Segmentation Benchmark](#real-data-segmentation-benchmark-bbbc039) and [Explainability and Calibration](#explainability-and-calibration-blood-cell-models).

**Limitations.** This is an educational project, not a medical device. Rare cell types (basophils, immature granulocytes) are classified less reliably than common ones (macro-F1 0.79 vs. 95.8% accuracy); the review flag catches only part of the errors; the smear detector skips most platelets; and photos that differ strongly from microscope scans (heavily edited or compressed web images) can still be misclassified.

## Features

- **Blood-cell classification that holds up on another scanner** — the app uses **DinoBloom-S**, a hematology foundation model, with a linear head trained on BloodMNIST at 224 px; tested on a second dataset from a different laboratory and scanner, it reaches 95.8% where models that score 96–99% on BloodMNIST fall to 65–76%, and it keeps 93.9% under strong simulated colour shifts. A PyTorch CNN trained from scratch (96.7% on BloodMNIST, Optuna-tuned) is shown alongside as the small baseline.
- **Blood-smear differential count** — a smear upload is split into one crop per stained cell; each is classified, and cells below 50% confidence are flagged for review.
- **Real-data segmentation benchmark** — every method evaluated on BBBC039, 200 fluorescence images of U2OS nuclei with expert annotations and an official train/validation/test split, using the metrics of the dataset's reference paper.
- **U-Net for nucleus segmentation** — a 1.9 M-parameter U-Net trained here on BBBC039 with the three-class (background / interior / boundary) formulation of Caicedo et al. (2019); available in the app's segmentation page.
- **Explainability (Grad-CAM)** — heatmaps of where each blood-cell model looked, verified against the `pytorch-grad-cam` library (correlation > 0.999).
- **Confidence calibration** — temperature scaling fitted on the validation split, with reliability diagrams and ECE on the test split; both models were already well calibrated (test ECE < 1%), so scaling changes their confidences only marginally.
- **FashionMNIST clothing classification** — a PyTorch CNN that predicts one of ten clothing classes from an uploaded image (92.8% test accuracy, up from 86.9% for the original fully connected network), with upload preprocessing that reproduces how the dataset was made.
- **Biomedical image segmentation** — six classical methods: Otsu thresholding, adaptive thresholding, K-Means, Canny edge filling, top-hat background correction, and a distance-transform watershed that splits touching cells — benchmarked against **Cellpose**, the standard deep-learning tool for cell segmentation. Brightfield images (dark cells on a light background, such as stained blood smears) are detected and inverted automatically.
- **Segmentation benchmark** — generates realistic synthetic microscopy images (touching nuclei, uneven illumination, blur, noise) with per-cell ground truth, and scores every method with pixel-level Dice / IoU and object-level instance F1.
- **Machine-learning utilities** — decision trees, random forests, ANOVA feature selection, Optuna tuning, and K-Means clustering analysis.
- **Image-processing fundamentals** — a NumPy implementation of 2D convolution with edge-detection, sharpening, Sobel, and box-blur kernels.
- **Visualization utilities** — EDA plotting helpers built with Matplotlib and Seaborn.

## Project Structure

```text
.
├── app.py                          # Streamlit home page
├── pages/
│   ├── 1_Deep_Learning.py           # FashionMNIST classification page
│   ├── 2_Biomedical_Segmentation.py # Biomedical segmentation page
│   └── 3_Blood_Cell_Classifier.py   # BloodMNIST classification page
├── src/
│   ├── vision_cnn.py                # PyTorch model and training helpers
│   ├── medical_cnn.py               # BloodMNIST CNN, training and inference helpers
│   ├── segmentation.py              # Classical segmentation, Dice / IoU / instance-F1 benchmark
│   ├── classification.py            # Supervised learning and Optuna tuning
│   ├── clustering.py                # K-Means and silhouette-based model selection
│   ├── convolution.py               # NumPy 2D convolution
│   ├── image_io.py                  # Upload loading: 16-bit, transparency, EXIF, size limits
│   ├── bbbc039.py                   # BBBC039 loader (mask decoding, percentile normalization)
│   ├── unet.py                      # U-Net, 3-class encoding/decoding for touching nuclei
│   ├── explain.py                   # Grad-CAM for the blood-cell models
│   ├── calibration.py               # Temperature scaling, ECE, reliability curves
│   ├── runtime.py                   # App resource settings: heavy-work lock, torch threads
│   ├── blood_cells.py               # Finds/crops stained cells, re-frames zoomed-out cells, review flag
│   ├── color_shift.py               # Colour-change suite for robustness tests; Reinhard normalization
│   └── visualization.py             # EDA visualization helpers
├── notebooks/demo_end_to_end.ipynb  # Walkthrough of the classical-ML, clustering, segmentation and convolution modules
├── public/                          # Figures, BloodMNIST and FashionMNIST sample images
├── train.py                         # FashionMNIST training script (CNN, or --model mlp)
├── train_medical.py                 # BloodMNIST training / tuning script
├── benchmark_segmentation.py        # Segmentation benchmark over many images
├── benchmark_bbbc039.py             # Real-data segmentation benchmark (BBBC039)
├── train_unet.py                    # Train the nucleus U-Net on BBBC039
├── calibrate_medical.py             # Temperature-scale the blood-cell models
├── probe_foundation.py              # Linear probes of DinoBloom-S / DINOv2 on BloodMNIST-224; exports the app model
├── eval_external.py                 # Cross-source evaluation on KU-Optofil PBC (other lab and scanner)
├── biomedical_dinobloom_s.pth       # DinoBloom-S backbone + trained head (86.6 MB, used by the app)
├── foundation_probe_results.json    # Probe results; external_results.json: cross-source results
├── unet_bbbc039.pth                 # Trained U-Net weights (7.8 MB) + *_results.json
├── bbbc039_results.json             # BBBC039 test-set results of every method
├── biomedical_*_calibration.json    # Fitted temperatures and before/after test metrics
├── fashion_cnn.pth                  # Trained FashionMNIST CNN (used by the app)
├── fashion_model.pth                # FashionMNIST fully connected baseline
├── biomedical_cnn.pth               # Trained BloodMNIST CNN weights
├── biomedical_cnn_results.json      # Hyperparameters and val/test accuracy
├── biomedical_resnet18.pth          # Fine-tuned ResNet-18 weights (reference model; 28 px upsampled)
├── biomedical_resnet18_224_results.json # ResNet-18 at native 224 px (weights not shipped; see below)
├── biomedical_resnet18_results.json # Its hyperparameters and val/test accuracy
├── requirements.txt                 # App dependencies (what Streamlit Cloud installs)
├── requirements-ml.txt              # scikit-learn, seaborn, Optuna
├── requirements-dev.txt             # ml + Cellpose, MedMNIST, pytest
├── .streamlit/config.toml           # Upload size cap, error-detail level
├── Dockerfile                       # Container for the Streamlit app (CPU-only PyTorch)
├── pyproject.toml                   # pytest configuration
├── CLAUDE.md                        # Contributor notes, review checklist and log
├── .github/workflows/tests.yml      # CI on Python 3.12: full, and app dependencies as deployed
└── tests/                           # pytest suite (run in CI on every push)
```

## Requirements

- Python 3.10 or later for the app (3.10–3.12 if you also want the Cellpose benchmark)
- A virtual environment is recommended

```bash
git clone https://github.com/jia-jing-zhang-dev/AI-Vision-Biomedical-Analytics-Platform.git
cd AI-Vision-Biomedical-Analytics-Platform
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate.bat       # Windows cmd
.\.venv\Scripts\Activate.ps1     # Windows PowerShell (may need: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
```

The dependencies are split into three files:

| File | Contents | Used by |
|---|---|---|
| `requirements.txt` | App dependencies, with CPU-only PyTorch | Streamlit Cloud, Docker |
| `requirements-ml.txt` | scikit-learn, seaborn, Optuna (classical ML, tuning) | Training, notebook, tests |
| `requirements-dev.txt` | `requirements-ml.txt` + Cellpose, MedMNIST, pytest | Training, benchmarks, tests |

```bash
pip install -r requirements.txt                                 # run the app
pip install -r requirements.txt -r requirements-ml.txt pytest   # app + tests (any Python version, no Cellpose)
pip install -r requirements.txt -r requirements-dev.txt         # everything (Python 3.10-3.12)
pip install notebook                                            # optional: to open the notebook
```

> Cellpose 3 requires `numpy<2.1`, which has no prebuilt wheels for Python 3.14, so use Python 3.10–3.12 for the extras. The deployed app does not install Cellpose and hides it automatically; its benchmark results are reported below.

> `requirements.txt` installs CPU-only PyTorch: it points `--find-links` at the PyTorch CPU wheels (deliberately not an extra package index, which would also serve outdated versions of other packages). To train on a GPU, first install the CUDA build of the torch/torchvision versions pinned in `requirements.txt`, using the command from the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

## Run the Web Application

From the project root, run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, then use the sidebar to switch between modules:

1. **Deep Learning** — upload a JPG, JPEG, or PNG clothing image and run FashionMNIST classification.
2. **Biomedical Segmentation** — either generate a synthetic benchmark image (realistic cells or simple shapes) to compare masks, Dice / IoU and instance F1, run a 10-image benchmark, or upload a grayscale or color medical image to view segmentation results from every algorithm.
3. **Blood Cell Classifier** — pick a bundled BloodMNIST test image (one per class) or upload your own, and see both models' predictions (custom CNN and ResNet-18) and class probabilities. An uploaded smear with several stained cells is split into one crop per cell, each crop is classified, and a differential count marks uncertain cells for review.

Uploads (JPG/PNG, including 16-bit and transparent PNGs and rotated phone photos) are accepted up to 25 megapixels and downscaled to at most 1024 px before processing, which keeps the app within Streamlit Cloud's memory limit.

### Run with Docker

```bash
docker build -t vision-analytics .
docker run -p 8501:8501 vision-analytics
```

Then open <http://localhost:8501>.

## Train Models

### FashionMNIST Clothing Classifier

```bash
python train.py               # CNN -> fashion_cnn.pth, fashion_cnn_results.json (~5 min on a laptop GPU)
python train.py --model mlp   # original fully connected baseline -> fashion_model.pth
```

The CNN (5 convolutions with batch normalization, global average pooling; 140 k parameters) trains for 25 epochs on 55,000 training images augmented with flips, shifts, scaling and small rotations. The epoch with the best accuracy on the remaining 5,000 training images (validation) is kept and the test set is evaluated once. The fully connected baseline keeps its original recipe (5 epochs, no augmentation). Both are deterministic for a given `--seed` on the same hardware.

| FashionMNIST, 10,000 test images | Fully connected (670 k parameters) | **CNN** (140 k parameters) |
|---|---|---|
| Test accuracy | 86.9% | **92.8%** |
| Test set, every image mirrored | 63.5% (ankle boots 0%) | **92.5%** (ankle boots 96%) |
| 2,000 photo-like uploads (small, off-centre, off-white background, half mirrored) | 85.8% | **90.5%** |

The CNN is right on 853 test images the baseline gets wrong and wrong on 267 it gets right (McNemar p < 10⁻⁷⁰). Its filters respond to an edge or a heel wherever it appears, and the augmentation teaches it that a mirrored or shifted item is the same item; the fully connected network ties every weight to one pixel position. The page shows the CNN's prediction and, for comparison, the baseline's.

### BloodMNIST Biomedical Classifier

```bash
python train_medical.py --optuna 20   # reproduces the shipped model (96.7%)
```

The script downloads BloodMNIST from MedMNIST, runs a 20-trial Optuna search on the validation split, trains the chosen CNN for 15 epochs, keeps the epoch with the best **validation** accuracy, evaluates the held-out **test** split once, and writes `biomedical_cnn.pth`, `biomedical_cnn_results.json` and `public/bloodmnist-confusion.png`. The Blood Cell Classifier page loads these files.

> Without `--optuna`, the default hyperparameters are used (96.0%) and the shipped weights, results and figure are **overwritten**; the old calibration file is removed (re-run `calibrate_medical.py`). Pass `--out runs/model.pth` (create `runs/` first) to write an experiment elsewhere; its results and figure then go next to it.

```bash
python train_medical.py --model mlp   # reproduce the original fully connected baseline
python train_medical.py --model resnet18   # fine-tune ImageNet-pretrained ResNet-18 (224×224 input, 10 epochs by default)
```

#### Results (BloodMNIST test split, 3,421 images)

| Model | Test accuracy |
|---|---|
| Fully connected baseline (flattened pixels → 128 → 8, 5 epochs) | 72.9% |
| CNN, default hyperparameters (seeds 0, 1, 2) | 96.0% / 96.4% / 96.4% |
| **CNN, Optuna-tuned (20 trials, seed 0) — shipped model** | **96.7%** |
| ResNet-18, ImageNet-pretrained, fine-tuned at 224×224 (torchvision) | 97.9% |
| *Published (MedMNIST v2, Table 3):* ResNet-18 (28) / ResNet-18 (224) / Google AutoML Vision | *95.8% / 96.3% / 96.6%* |

The jump from 72.9% comes with switching to a CNN trained longer (15 instead of 5 epochs, AdamW with a cosine schedule) with augmentation (random flips and 90° rotations); no ablation separates architecture, augmentation and training length. Optuna tuning adds 0.3–0.7 points over the three default runs. On the same test images the tuned model beats the seed-0 default (paired McNemar test, 47 vs. 25 images only one model gets right, p = 0.01), but the tuned configuration was trained with a single seed and the default varies by 0.4 points between seeds, so the tuning gain is small and not firmly established. The 1.3-point gain of ResNet-18 over the custom CNN is significant (McNemar, 63 vs. 20, p < 0.001).

Training is deterministic (`set_deterministic` fixes every seed and forces deterministic GPU kernels), so each command reproduces its number exactly on the same hardware; other GPUs or library versions can differ by a few tenths of a point.

**Custom CNN vs. a mature pretrained model:** the custom CNN (2.2 MB, 0.56 M parameters) is on par with the best published BloodMNIST results in the MedMNIST v2 paper. Fine-tuning ImageNet-pretrained ResNet-18 (44.8 MB, 11.2 M parameters, 20× larger) gains another 1.3 points (97.92% vs. 96.67%) — transfer learning helps, at a large cost in model size. The published baselines were trained under the paper's own protocol (not with this project's augmentation, tuning or ImageNet pretraining), so treat that comparison as indicative rather than like-for-like. The app shows both models' predictions side by side.

![BloodMNIST confusion matrix](public/bloodmnist-confusion.png)

The hardest classes are monocytes (91.5% recall, most often mistaken for immature granulocytes) and immature granulocytes (92.2% recall, most often mistaken for monocytes).

*Data: [MedMNIST v2](https://medmnist.com/) BloodMNIST, CC BY 4.0 (Yang et al., Scientific Data 2023). Educational project — not for clinical use.*

## Segmentation Benchmark

```bash
python benchmark_segmentation.py --images 50   # also refreshes public/cell-segmentation-benchmark.png when Cellpose is installed
```

Without Cellpose the shipped figure is left untouched (it has a Cellpose panel); use `--figure path.png` to save an example elsewhere.

Results on 50 synthetic cell images (~45 nuclei each, mean ± std):

| Method | Dice | IoU | Instance F1 |
|---|---|---|---|
| *Cellpose 3, pretrained `nuclei` model (deep-learning reference)* | *0.979 ± 0.013* | *0.960 ± 0.023* | *0.895 ± 0.049* |
| **Top-hat + watershed** | **0.935 ± 0.017** | **0.879 ± 0.030** | **0.798 ± 0.060** |
| Top-hat + Otsu | 0.935 ± 0.017 | 0.879 ± 0.030 | 0.609 ± 0.082 |
| Otsu | 0.932 ± 0.018 | 0.873 ± 0.032 | 0.598 ± 0.084 |
| K-Means (K=2) | 0.932 ± 0.018 | 0.873 ± 0.031 | 0.598 ± 0.084 |
| Adaptive threshold | 0.774 ± 0.027 | 0.632 ± 0.035 | 0.521 ± 0.079 |
| Canny + hole filling | 0.823 ± 0.014 | 0.700 ± 0.021 | 0.389 ± 0.075 |

![Segmentation benchmark example](public/cell-segmentation-benchmark.png)

On a real brightfield blood smear, the light background is detected and the image inverted before segmentation:

![Brightfield example](public/brightfield-demo.png)

**What this shows:** pixel-level Dice cannot tell the thresholding methods apart — they all find the foreground well. Instance F1 can: thresholding merges touching nuclei into one object, while the watershed, seeded at peaks of the distance transform, separates them (+0.20 instance F1). Top-hat illumination correction helps only slightly here, because Otsu is already fairly robust to this level of shading. Cellpose, a pretrained deep-learning model (run with automatic diameter estimation, so nothing is tuned to these images), is the strongest method — about 0.10 instance F1 above the watershed — and shows how far classical methods are from the state of the art. Instance F1 counts a predicted object as correct when it overlaps a true cell with IoU > 0.5; objects under 20 pixels are ignored in both the prediction and the ground truth.

Method parameters were chosen while looking at images 0–9. On 50 unseen images (seeds 100–149) the result holds: Cellpose 0.904, watershed 0.819, Otsu 0.620 instance F1.

The original three-shape image is still available as an "easy" sanity check, where Otsu, K-Means, top-hat and watershed all reach Dice 1.0.

## Real-Data Segmentation Benchmark (BBBC039)

[BBBC039](https://bbbc.broadinstitute.org/BBBC039) contains 200 fluorescence images (520×696, 12-bit) of U2OS cell nuclei with expert annotations of every nucleus (~23,000 in total) and an official split of 100 training / 50 validation / 50 test images. Download `images.zip`, `masks.zip` and `metadata.zip` from the dataset page and unzip them into `data/bbbc039/`.

```bash
python train_unet.py --data-dir data/bbbc039                                     # ~5 min on a laptop GPU
python benchmark_bbbc039.py --data-dir data/bbbc039 --unet unet_bbbc039.pth       # evaluates the 50 test images
```

Images are converted to 8-bit with percentile normalization (0.5–99.8%). Metrics follow the dataset's reference paper (Caicedo et al., *Cytometry A* 2019): per-image instance F1 at IoU > 0.5 and **average F1** over IoU thresholds 0.50–0.90 (step 0.05), which also rewards accurate outlines, averaged over the 50 test images. One difference: the main columns ignore objects under 20 pixels on both sides (156 of the 5,720 annotated test nuclei, mostly micronuclei and fragments), whereas the paper keeps every annotated nucleus. The last column uses the paper's protocol (all annotations kept, predictions still filtered); compare the paper's row with that column.

| Method (BBBC039 test set, 50 images) | Dice | Instance F1 @ IoU 0.5 | Average F1, IoU 0.50–0.90 | Average F1, paper protocol |
|---|---|---|---|---|
| **U-Net (trained here)** | 0.951 ± 0.009 | 0.951 ± 0.077 | 0.863 ± 0.100 | 0.850 |
| Cellpose 3 (pretrained) | 0.948 ± 0.124 | 0.938 ± 0.148 | 0.902 ± 0.156 | 0.889 |
| Adaptive threshold | 0.957 ± 0.009 | 0.898 ± 0.087 | 0.830 ± 0.101 | 0.819 |
| Top-hat + watershed | 0.947 ± 0.014 | 0.900 ± 0.090 | 0.805 ± 0.103 | 0.794 |
| Otsu | 0.950 ± 0.013 | 0.829 ± 0.107 | 0.744 ± 0.107 | 0.733 |
| Top-hat + Otsu | 0.946 ± 0.014 | 0.826 ± 0.108 | 0.734 ± 0.107 | 0.723 |
| Canny + hole filling | 0.868 ± 0.017 | 0.663 ± 0.130 | 0.431 ± 0.099 | 0.425 |
| *Reference paper: U-Net / DeepCell / CellProfiler (advanced)* | | | | *0.898 / 0.858 / 0.811* |

![BBBC039 example](public/bbbc039-segmentation.png)

**What this shows:**
- The U-Net trained here is on par with Cellpose at IoU 0.5 (0.951 vs. 0.938 on average). Cellpose scores higher on 35 of the 50 images (U-Net higher on 12, 3 ties); the U-Net's higher mean comes from one sparse image, `IXMtest_F12_s8`, where Cellpose returns a single object for 7 nuclei (F1 0.0). Without that image Cellpose leads by 0.008 at IoU 0.5 and by 0.058 in average F1. Under the paper's protocol the U-Net (0.850) beats the paper's CellProfiler baseline (0.811) on average F1, but its outlines are less precise than Cellpose's or the paper's own U-Net (0.889 and 0.898). It is a small model (1.9 M parameters, 60 short epochs) without the paper's training budget.
- Real data reorders the classical methods: the adaptive threshold, weak on the synthetic benchmark, is the best classical method on real nuclei by average F1 and Dice (0.830 / 0.957) and ties with the watershed at IoU 0.5 (0.898 vs. 0.900), while Canny edges fail on the soft nucleus borders. These classical scores depend on the 20-pixel filter on predictions: without it the adaptive threshold's average F1 falls to about 0.66.
- The classical methods use the parameters developed on the synthetic benchmark, without tuning on BBBC039.
- The large standard deviations come mostly from one hard test image (`IXMtest_A12_s7`) on which every method struggles (F1: U-Net 0.45, adaptive 0.35, watershed 0.32); without it the U-Net's F1 standard deviation is 0.028 instead of 0.077. Cellpose's large standard deviation comes from `IXMtest_F12_s8` (above).

**Protocol note.** The U-Net checkpoint was selected on the validation split. Its first test-set evaluation exposed a decoding error: the boundary class extends one pixel outside each nucleus (as in the paper), but decoding grew nuclei over that pixel, so every nucleus came out one pixel too large (test average F1 0.622). The fix (erode the predicted foreground by one pixel) follows from the label definition; it was compared with the old decoding on the **validation** split only (average F1 0.635 → 0.854), and the test set was then evaluated once more. A unit test now checks that encoding and decoding reproduce nucleus outlines. A later code review found that the same erosion also trimmed nuclei cut by the image border; fixing it raised validation average F1 from 0.854 to 0.873, and the test set was evaluated a third time with that fix (0.845 → 0.863). That second fix was validated on the validation split, but the search for it was not blind to the earlier test results. The weights never changed; only the decoding did. Large uploads are processed in 384-pixel tiles, which keeps memory bounded without changing results on BBBC039-sized images.

### Blood-smear uploads

The models were trained on BloodMNIST crops showing one centred cell, so a whole smear gives a meaningless answer (an early upload with several neutrophils came out as "immature granulocyte"). `src/blood_cells.py` finds the purple-stained cells (blue minus green; red cells and background are not stained), merges the lobes of each nucleus, and crops every cell the way BloodMNIST frames them. The stained region spans a median 0.60 of a BloodMNIST crop (measured on the validation split; lymphocytes 0.50, monocytes 0.75); because size helps tell the classes apart, one crop size is used for all cells of an image, derived from their median size.

Checked on smear-like mosaics of 3×3 BloodMNIST test cells (100 mosaics, 900 cells, scaled 2–5×; settings fixed on validation mosaics first):

| | Custom CNN | ResNet-18 |
|---|---|---|
| Original single-cell crops (reference) | 95.7% | 97.3% |
| Cells found, cropped and classified by the pipeline | 86.4% | 89.7% |

99.0% of leukocytes and erythroblasts were found (777/785); platelets mostly were not (12/115), because small stained objects next to larger cells are skipped. Test-time augmentation (flips/rotations, three crop sizes) did not help on validation. The mosaics have hard seams between tiles, so real smears may behave differently; uploads with only one stained cell are classified as a whole image, as before (6 of 500 single-cell test crops were mistaken for smears).

The page then reports a **differential count** (count of each cell type, and shares once at least 10 cells are counted) like a lab report. Cells are now classified by DinoBloom-S at 224 px; one below 50% confidence is marked **needs review** (red box). On smear mosaics of BloodMNIST-224 test cells (100 mosaics, 900 cells) 97.9% of leukocytes and erythroblasts were found, DinoBloom-S classified 99.0% of the found cells correctly, and the review rule kept 99.6% of them at 99.1% accuracy. (An earlier version used the 28 px CNN and ResNet-18 with a two-model agreement rule: 86.4% / 89.7% on the same kind of mosaics; on one smear of neutrophils from another source only 3 and 6 of 9 cells were right, which started the cross-scanner work below.)

## Higher Resolution, Foundation Models and a Second Scanner (blood cells)

BloodMNIST's 28×28 images lose most of a cell's detail, and every published model scores 96–99% on its own test set. The harder question is whether a model still works on images from another hospital. This section answers it in three steps; every setting was chosen on validation data and each test set was evaluated once.

**1. Native 224×224 images** (MedMNIST+, Doerrich et al., *Sci Rep* 2025, same 17,092 cells and split): ResNet-18 fine-tuned on 224 px images reaches **99.12%**, against 97.92% when it reads 28 px images upsampled to 224 (McNemar p = 2×10⁻⁷), in line with the 98.94% ± 0.08 MedMNIST+ reports.

```bash
python train_medical.py --model resnet18_224 --data-dir DIR   # DIR holds bloodmnist_224.npz (MedMNIST+, 1.5 GB)
```

**2. Foundation models instead of training from scratch.** `probe_foundation.py` freezes a pretrained vision transformer and trains only a logistic-regression head on its embeddings (regularization chosen on validation):

| BloodMNIST-224 test (3,421 images) | Accuracy |
|---|---|
| ResNet-18, fine-tuned end to end | 99.12% |
| **DinoBloom-S** (hematology foundation model, 22 M parameters), linear head | **98.95%** |
| DINOv2-B (general, 86 M parameters), linear head | 98.07% |
| DINOv2-S (general, 22 M parameters), linear head | 97.75% |

DinoBloom-S ties the fine-tuned ResNet (p = 0.39) and beats both general-purpose DINOv2 models (p ≤ 0.0002). It was pretrained on 380,000 white-cell images from 13 datasets, with Acevedo et al. (BloodMNIST's source) held out (Koch et al., MICCAI 2024).

**3. Another laboratory and scanner.** `eval_external.py` tests every model, without retraining, on **KU-Optofil PBC** (Koç University, 2025; Sysmex DI-60, May-Grünwald-Giemsa; CC BY 4.0), whereas BloodMNIST comes from a CellaVision DM96 at the Hospital Clinic of Barcelona (Acevedo et al. 2020; also May-Grünwald-Giemsa, so the stain type is the same but the lab protocol, smears and scanner differ). It was published after DinoBloom, so DinoBloom cannot have seen it. Its classes are mapped to BloodMNIST's (segmented and band neutrophils → neutrophil; myelocytes and metamyelocytes → immature granulocyte); giant platelets, platelet clusters, blasts and reactive lymphocytes have no counterpart and are left out. The dataset's own patient-level split is used: validation patients to choose settings, test patients (4,949 cells, 55 patients) reported once, with 95% intervals from resampling patients.

The first run exposed a framing difference: this scanner shows each cell at about half the size (stained region 0.36 of the frame vs. 0.66 on BloodMNIST-224), and the models trained on BloodMNIST collapsed (ResNet-18 at 224 px called 96% of the cells "erythroblast"). Zoomed-out cells are now re-framed around the stained cell to BloodMNIST's scale first (`reframe_single_cell`; only when the cell spans under 0.45 of the frame, so BloodMNIST-framed images are untouched; both values chosen on validation). Estimating magnification from red-cell size instead was tried on validation and dropped (red cells too few and too variable in BloodMNIST frames).

| KU-Optofil test patients | As delivered | **Re-framed** | Macro-F1 (re-framed) |
|---|---|---|---|
| Custom CNN (28 px) | 4.2% | 65.0% | 0.33 |
| ResNet-18 (28 px upsampled) | 2.7% | 63.7% | 0.39 |
| ResNet-18 (224 px) | 3.5% | 76.2% | 0.58 |
| **DinoBloom-S + linear head** | 81.6% | **93.8%** [91.7, 95.4] | **0.77** |
| DINOv2-S + linear head | 57.7% | 79.6% | 0.61 |
| DINOv2-B + linear head | 74.5% | 86.0% | 0.59 |

The model that ties for best on BloodMNIST (ResNet-18 at 224 px) is 18 points behind DinoBloom-S on the new scanner, which matches a recent study of frozen hematology encoders under acquisition shift (Sharma & Tapadiya, HemaRAI 2026 workshop, arXiv 2608.25148). Macro-F1 stays well below accuracy because rare classes (basophils, immature granulocytes, erythroblasts) are harder; the external set's neutrophils dominate accuracy.

**4. Colour robustness.** Photos from other labs differ in white balance, stain strength, exposure and compression. `src/color_shift.py` applies a seeded suite of such changes (evaluation only) at two severities. Two remedies were compared on the validation sets: Reinhard colour normalization to BloodMNIST's CIELAB statistics, which hurt everywhere (BloodMNIST validation 99.3% → 89.0%: per-image statistics depend on how much of the frame is cell, and normalizing them erased colour cues), and training the head on the BloodMNIST images plus one colour-jittered copy each (`--color-jitter`), which helped. The app ships the jittered head (BloodMNIST test 98.89% vs. 98.95% without, i.e. unchanged):

| KU-Optofil test patients (re-framed) | No colour change | Mild changes | Strong changes |
|---|---|---|---|
| **DinoBloom-S, colour-jittered head (app)** | **95.8%** [94.2, 97.1], F1 0.79 | 95.2% | 93.9% |
| DinoBloom-S, plain head | 93.8% | 92.9% | 90.7% |
| ResNet-18 (224 px) | 76.2% | 67.7% | 42.7% |
| Custom CNN (28 px) | 65.0% | 34.5% | 12.4% |

The app model is exported with its head as one checkpoint (`python probe_foundation.py --data-dir DIR --models dinobloom_s --color-jitter --export biomedical_dinobloom_s.pth`; loads offline). Cells it classifies with less than 50% confidence are flagged for review; the threshold was chosen on the validation patients as the lowest one whose accepted cells are at least 98% correct. On the test patients it accepted 98.1% of cells at 96.8% accuracy, below that target, and flagged only 55 of the 209 errors: the jittered head's confidences are softer, so the flag catches far fewer mistakes than with the plain head (57% at a 0.70 threshold). Counted cells can still be wrong.

## Explainability and Calibration (blood-cell models)

**Grad-CAM.** For DinoBloom-S (a vision transformer) the activations are the patch tokens entering the last block's attention, the layer `pytorch-grad-cam` uses for ViTs; its heatmaps match the library's (correlation 1.000). The Blood Cell Classifier page can show a Grad-CAM heatmap for each model: the image regions that pushed it towards its prediction. The implementation (`src/explain.py`) splits the forward pass by hand instead of registering hooks, because the app shares one cached model between all visitors. A test checks that its heatmaps match the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library (correlation > 0.999).

**Calibration.** A model is well calibrated when, among predictions made with 90% confidence, about 90% are right. `calibrate_medical.py` fits a single temperature per model on the validation split (temperature scaling, Guo et al. 2017) and measures the expected calibration error (ECE, 15 bins, via `torchmetrics`) on the test split:

| Model | Temperature | Test ECE before → after | Test accuracy |
|---|---|---|---|
| Custom CNN | 0.89 | 0.89% → 0.65% | 96.67% (unchanged) |
| ResNet-18 | 1.09 | 0.50% → 0.36% | 97.92% (unchanged) |

Both models were already well calibrated (ECE below 1%). Temperature scaling changes very little, and the change is not statistically significant: a paired bootstrap over the test images (1,000 resamples) gives a 95% interval for the ECE change of -0.60 to +0.38 points for the CNN and -0.30 to +0.22 points for ResNet-18, both including zero. The fitted temperatures are applied in the app, but they are close to 1, so the confidences barely change.

![Reliability diagrams](public/bloodmnist-calibration.png)

## Run Tests

From the repository root, after installing at least `requirements.txt`, `requirements-ml.txt` and pytest:

```bash
pytest tests/
```

Tests that need Cellpose or MedMNIST are skipped when those packages are absent.

The tests cover classification, clustering, segmentation metrics, 2D convolution (checked against SciPy), and the BloodMNIST CNN. `tests/test_app.py` drives every page of the Streamlit app headlessly with Streamlit's `AppTest`: each button, selector and seed input, plus uploads of brightfield, fluorescence, oversized and blood-cell images.

## Troubleshooting

### `fashion_cnn.pth` or `fashion_model.pth` cannot be found

Run `python train.py` (CNN) or `python train.py --model mlp` (baseline) from the project root, or confirm that the weight files are still located there.

### Dataset download fails during training

Both `train.py` and `train_medical.py` download their datasets on the first run. Check your internet connection and try again.

### Uploaded clothing images produce weak predictions

The model is trained on 28×28 grayscale FashionMNIST images. For better results, use a clear, centered image of a single clothing item. The application prepares uploads the way FashionMNIST itself was made: it converts them to grayscale, inverts them when the background is light (FashionMNIST items are light on black), removes the background, crops and centres the item and scales its longer side to fill 28 pixels. All FashionMNIST shoes face the same way and the fully connected baseline never recognizes a mirrored ankle boot (0% on the test set), so left-heavy items are mirrored first (for the baseline, all-mirrored test set 63.5% → 86.5%); the CNN, trained with flips, does not depend on this. The page shows this 28×28 input and the top-3 probabilities. On 2,000 test images turned into product-photo-like uploads (small, off-centre, on an off-white background) this raised the baseline's accuracy from 12% (squeezing the whole photo to 28×28) to 86%; real photos with shadows or textured backgrounds remain harder.

## Data and Credits

- **BloodMNIST** — Yang et al., *MedMNIST v2*, Scientific Data 2023, built from Acevedo et al., *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief 2020; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `public/blood_samples/` holds the first test image of each class at 224×224 (MedMNIST+).
- **FashionMNIST** — Xiao, Rasul & Vollgraf (Zalando Research), arXiv:1708.07747, 2017, MIT License. `public/fashion_samples/` holds one unmodified 28×28 test image per class.
- **ResNet-18** — initialized from torchvision's ImageNet-pretrained weights before fine-tuning.
- **Cellpose** — Stringer et al., *Cellpose: a generalist algorithm for cellular segmentation*, Nature Methods 2021 (the `nuclei` model); the installed version is Cellpose 3 (Stringer & Pachitariu, Nature Methods 2025).
- **U-Net** — Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- **BBBC039** — Caicedo et al., *Evaluation of Deep Learning Strategies for Nucleus Segmentation in Fluorescence Images*, Cytometry A 2019; available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods 2012), CC0. The three-class U-Net formulation and the average-F1 metric follow this paper.
- **DinoBloom** — Koch et al., *DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology*, MICCAI 2024 (code Apache-2.0; timm conversion `1aurent/vit_small_patch14_224.dinobloom`). **DINOv2** — Oquab et al., 2023 (Apache-2.0). Loaded with **timm** (Wightman).
- **MedMNIST+** (224 px BloodMNIST) — Doerrich et al., *Rethinking model prototyping through the MedMNIST+ dataset collection*, Scientific Reports 2025; CC BY 4.0.
- **KU-Optofil PBC** — Yarıkan et al., *A large-scale peripheral blood cell dataset for automated hematological analysis*, Scientific Data 2026; Zenodo 10.5281/zenodo.17333317, CC BY 4.0. Used only for evaluation, not redistributed.
- **Grad-CAM** — Selvaraju et al., ICCV 2017. **Temperature scaling** — Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (fitting procedure after github.com/gpleiss/temperature_scaling).
