# CLAUDE.md

Guidance for Claude (and humans) working on this repository.

## Project

Streamlit app + reusable modules for biomedical image analysis, deployed at
<https://visionserve-full-stack-ml-framework.streamlit.app/>.

- `src/medical_cnn.py` — BloodMNIST classifiers: custom CNN (shipped, 96.7%), MLP baseline (72.9%), ImageNet-pretrained ResNet-18 reference (97.9%)
- `src/segmentation.py` — classical segmentation, synthetic cell benchmark, Dice/IoU/instance F1, optional Cellpose reference
- `src/vision_cnn.py` — FashionMNIST MLP (86.9%)
- `src/classification.py`, `src/clustering.py`, `src/convolution.py`, `src/visualization.py` — ML utilities
- `pages/` — Streamlit pages; `app.py` — home page
- `train.py`, `train_medical.py`, `benchmark_segmentation.py` — reproducible scripts behind every number in the README

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt   # dev extras need Python <= 3.12 (Cellpose)
pytest tests/                                              # unit + Streamlit AppTest end-to-end tests
python train_medical.py [--model cnn|mlp|resnet18] [--optuna N]
python benchmark_segmentation.py --images 50 [--start-seed 100]
streamlit run app.py
```

## Facts that are easy to get wrong

- **Streamlit Cloud runs Python 3.14** and ignores `runtime.txt`. Anything in
  `requirements.txt` must install on 3.14 from prebuilt wheels.
- **Cellpose 3 pins `numpy<2.1`** (no 3.14 wheels) → it lives in
  `requirements-dev.txt` only; the app hides Cellpose when it is not installed.
- CI (`.github/workflows/tests.yml`) runs Python 3.12 with dev extras **and**
  Python 3.14 with exactly the deployment's dependencies.
- After every push the deployed app must be rebooted by hand in Streamlit Cloud
  (the deployment is not auto-syncing).
- The user's Windows checkout lives under a very long path: datasets need the
  `\\?\` long-path prefix (see how `--data-dir` is passed), and `uv`/venvs in the
  scratch folder break. Use the global Python for local checks.
- Training is deterministic via `set_deterministic(seed)`; every README number
  must come from a committed script run. After retraining, re-check every
  hard-coded number (README, `app.py`, pages, tests, this file).
- BloodMNIST uploads must be reduced to 28×28 with **BOX** (area) resampling;
  bicubic flips predictions.
- Brightfield images (light background) are inverted before segmentation
  (`has_bright_background`).
- Label images are `int32`; binary masks are any other dtype. `label_objects`
  and `evaluate` both rely on this convention.

## Review checklist — use ALL of it on every check

Approach every review as if nothing has been checked before. Do not skip an
item because an earlier round covered it; re-verify against the current code.

1. **Correctness** — read every changed/central function line by line: index
   order, off-by-one, dtype/shape assumptions, empty inputs, edge cases
   (0 objects, 1 object, non-square, huge images, grayscale/RGBA/16-bit).
2. **ML methodology** — no test-set leakage (tuning only on train/val or CV),
   seeds fixed, metrics computed the way the README says.
3. **Claims vs. code** — every number and sentence in README, `app.py` and the
   pages must be reproducible from the code in this repo; re-run the relevant
   script when in doubt. Cite published numbers only from the source.
4. **App** — every page, button, selector and upload path works
   (`tests/test_app.py`); no deprecated Streamlit APIs; memory-safe on
   Streamlit Cloud (~1 GB).
5. **Dependencies & deployment** — installs on Python 3.14 from wheels; CI
   green on both Python versions; versions pinned where breakage is likely.
6. **Tests** — every fixed bug gets a regression test; tests assert something
   meaningful (no always-true assertions).
7. **Code quality** — no duplicated helpers, no dead code, docstrings match
   behaviour, CLI rejects options it would ignore.
8. **Repo hygiene** — no committed data, caches, OS junk or secrets; README
   links and images resolve.

Every finding must be **verified** (reproduced or proven from the code) before
it is fixed. After fixing: run `pytest tests/`, then log the round below.

## Review log

Pre-loop audits (2026-10-13) fixed: brightfield segmentation inverted;
convolution crashed on non-square images; the sklearn Optuna helper tuned on
the test set; SVC without feature scaling; upload memory
blow-up; deprecated `use_container_width`; seaborn palette deprecation;
Cellpose blocking the Python 3.14 deployment; `evaluate()` guessing the
ground-truth type from its max value; `fit()` crash when validation never
improves; CLI silently ignoring `--optuna`/`--epochs`; duplicated
`color_labels`.

### Review loop (10 rounds)

| Round | Extra focus | Verified findings | Fixed | Tests |
|---|---|---|---|---|
| 1 | Core-module correctness | 7 verified: adaptive threshold sign (background as foreground); 16-bit uploads saturated to white; FashionMNIST page always inverted (8.8% on its own test images); Cellpose empty on large cells; unseeded DecisionTree/silhouette/train.py; silhouette in wrong feature space; degenerate inputs (empty-mask Dice 0 vs F1 1, flat-image K-Means, tiny images, convolve2d/eda_report/KMeansExplorer validation) | All 7; added `src/image_io.py`; README Adaptive row re-measured | 29 → 44 |
| 2 | ML methodology & reproducibility | 10 verified: CNN size claim wrong (2.2 MB / 20×, not 0.58 MB / 77×); GPU training non-deterministic (same seed gave 95.97–96.26%); train.py no longer reproduced 88.1%; ResNet command ran 15 instead of 10 epochs; default-CNN range and "Optuna within noise" claim unsupported; `--out` without .pth overwrote the weights and any run overwrote the README figure; `.dockerignore` lines fused; stale test count / structure; GT cells not size-filtered like predictions; `compare_classifiers` ranked by test accuracy | All 10; added `set_deterministic` (full rerun matches to the last digit); retrained every model and re-measured every number (CNN 96.7%, ResNet 97.9%, FashionMNIST 86.9%, instance F1 0.80 / 0.60 / Cellpose 0.90); MedMNIST comparison caveated | 44 → 44 (2 reworked) |
| 3 | Streamlit app | 6 verified: 16-bit uploads crashed the previews on pages 1 and 3; large uploads could exceed ~1 GB (8000×8000 16-bit → 1.8 GB); corrupt/truncated files showed raw tracebacks; EXIF orientation ignored (rotated phone photo misclassified); ground-truth cell count in title ≠ table; flat images flagged as brightfield and fully foreground in Otsu | All 6 via one shared `load_upload` (validation, EXIF, 25 MP limit, downscale before conversion; worst case now ~200 MB); clearer Cellpose text on home page; guard for missing weights | 44 → 62 |
