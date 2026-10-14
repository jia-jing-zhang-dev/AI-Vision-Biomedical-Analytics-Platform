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
# requirements.txt = app only (Streamlit Cloud); requirements-ml.txt = sklearn/seaborn/optuna;
# requirements-dev.txt = ml + cellpose/medmnist/pytest
pytest tests/                                              # unit + Streamlit AppTest end-to-end tests
python train_medical.py [--model cnn|mlp|resnet18] [--optuna N]
python benchmark_segmentation.py --images 50 [--start-seed 100]
streamlit run app.py
```

## Facts that are easy to get wrong

- **Streamlit Cloud runs Python 3.14** and ignores `runtime.txt`. Anything in
  `requirements.txt` must install on 3.14 from prebuilt wheels.
- `requirements.txt` points `--find-links` at the PyTorch **CPU** wheels.
  Without it Linux installs CUDA torch + ~3 GB of nvidia packages (Streamlit
  Cloud did this); as `--extra-index-url` instead, uv also took outdated
  requests/urllib3/certifi from the PyTorch mirror. Torch/torchvision
  versions live only in `requirements.txt`.
- **Cellpose 3 pins `numpy<2.1`** (no 3.14 wheels) → it lives in
  `requirements-dev.txt` only; the app hides Cellpose when it is not installed.
- CI (`.github/workflows/tests.yml`) runs Python 3.12 with dev extras **and**
  Python 3.14 with the deployment's dependencies installed the same way (uv,
  wheels only), plus `requirements-ml.txt` and pytest for the tests.
- After every push the deployed app must be rebooted by hand in Streamlit Cloud
  (the deployment is not auto-syncing).
- The user's Windows checkout lives under a very long path: datasets need the
  `\\?\` long-path prefix, e.g.
  `python train_medical.py --data-dir "\\?\C:\full\path\to\data"` (in Python:
  `chr(92)*2 + "?" + chr(92) + os.path.realpath("../data")`), and `uv`/venvs in
  the scratch folder break. Use the global Python for local checks.
- `python train_medical.py` without `--optuna` overwrites the shipped CNN;
  experiments use `--out`.
- Training is deterministic via `set_deterministic(seed)`; every README number
  must come from a committed script run. After retraining, re-check every
  hard-coded number (README, `app.py`, pages, tests, this file).
- BloodMNIST uploads must be reduced to 28×28 with **BOX** (area) resampling;
  bicubic flips predictions.
- Brightfield images (light background) are inverted before segmentation
  (`has_bright_background`).
- Label images are any integer dtype except uint8 (int32/int64/uint16);
  binary masks are bool or uint8. `is_label_image` defines this and both
  `label_objects` and `evaluate` use it. Segmenters take 2-D uint8 only.
- App pages resolve files from `Path(__file__)`, never the working directory.
- BBBC039 lives in `data/bbbc039/` (gitignored; CC0, ~80 MB from
  bbbc.broadinstitute.org/BBBC039). Use the official split: train the U-Net
  on training, select on validation, evaluate test once
  (`benchmark_bbbc039.py`). Report Dice, F1@0.5 and average F1 (IoU
  0.50–0.90, the paper's metric).
- U-Net classes follow Caicedo et al.: the boundary band includes one pixel
  outside each nucleus, so `classes_to_labels` erodes the foreground by one
  pixel (without it every nucleus is 1 px too large; test AF1 0.62 vs 0.85).
- `load_model` applies `<weights>_calibration.json` (temperature from
  `calibrate_medical.py`); predictions are unchanged, confidences calibrated.
  `fit_temperature` needs the strong-Wolfe line search (it stopped early before).
- Grad-CAM (`src/explain.py`) is hook-free on purpose (models are shared across
  Streamlit sessions); a test compares it with pytorch-grad-cam (CI installs
  that with `--no-deps`, because it wants GUI OpenCV).
- Long paths also bite scripts: keep helper script names short or run them
  from a short temp path.

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
| 4 | Test-suite quality (mutation testing, ~60 mutants) | 11 verified gaps: FashionMNIST output never checked (random weights passed); earlier fixes without regression tests (Optuna test-set tuning, SVC scaling, CV ranking, `fit` best-epoch restore, `set_deterministic`, CLI checks, `--out` paths); benchmark numbers not pinned (watershed F1 0.81→0.73 undetected); `label_objects` size filter untested; metrics only tested at 0/1; convolve2d stride and single-axis flip undetected; `has_bright_background` border logic barely tested; `to_8bit` P/LA/flat-16-bit and `predict_proba` normalization untested; CI depends on cellpose.org; weak/redundant assertions; scripts untested | All 11: `tests/test_regressions.py` (27 tests), bundled FashionMNIST samples, `parse_args`/`output_paths` refactor, `--figure` for the benchmark script, CI cache for Cellpose models; the 11 surviving mutants re-run and now all caught | 62 → 89 |
| 5 | Dependencies, CI, deployment, Docker | 8 verified: Streamlit Cloud installed CUDA torch (~4.1 GB of nvidia/triton) and CI never tested that; `streamlit>=1.50` too loose (AppTest uploads need 1.56, guard was dead code); sklearn/seaborn/optuna installed on the deployment though unused by the app; `pillow` unbounded (needs 9.1); missing credits for FashionMNIST samples and BloodMNIST changes; Docker root user / no healthcheck / Python 3.12; CI actions outdated, no timeout/permissions, 3.14 job not wheels-only, torch versions in 3 places; dataset in git history | 7 of 8: CPU wheel index in requirements.txt (uv-verified: 0 nvidia packages on 3.12 and 3.14), `requirements-ml.txt`, `streamlit>=1.56`, `pillow>=9.1`, CI installs the 3.14 job like Streamlit Cloud (uv, wheels only, asserts +cpu), actions v7/v7/v6, Docker 3.14 non-root + healthcheck, README credits. Not done: rewriting git history (destructive); LICENSE and the brightfield photo's source need the owner's decision | 89 → 89 (Streamlit 1.65 verified) |
| 6 | Error handling & input validation | 12 verified: uint16/int64 label images silently treated as binary (45 cells → 28); `iou_thresh` < 0.5 gave F1 > 1; one NaN blanked a float image; unknown `arch` silently used CNN preprocessing; `instance_f1` accepted mismatched shapes; segmenters crashed deep in OpenCV (or wrapped uint16) on non-uint8 input; `predict_proba` device/array/0-px issues and unclear checkpoint errors (page 3 traceback on a corrupt file); `train.py` ignored all arguments (even `--help`); `train_medical` accepted `--epochs 0`, stray `--tune-epochs`, missing output folder; benchmark script validated nothing and failed only after the run; pages used working-directory-relative paths | All 12 (+ figures now closed); `is_label_image`, `_require_gray_uint8`, checkpoint validation, argparse for `train.py`, early CLI validation, `ROOT`-relative paths | 89 → 121 |
| 7 | Performance & resources (measured) | 4 verified: segmentation page spent ~90% of its time rendering matplotlib PNGs at dpi 200 (453 ms click, 425 ms of it drawing); label-id-proportional memory in `instance_f1`/`label_objects`/`color_labels` (+1.1 GB for uint16 ids up to 65k, +2.4 GB for id 1e8); Cellpose ran in page tests not about Cellpose (half the suite time); top-hat computed twice per `run_all` | All 4: `st.image` on arrays (click 453 → 147 ms), `_compact_ids` (id 1e8: 11 ms, no extra memory), uint8 `color_labels`, `@pytest.mark.cellpose` opt-in (suite 64 → 38 s), watershed reuses the top-hat mask (identical output) | 121 → 121 |
| 8 | Security & repo hygiene | 10 verified: rounds 1–7 not pushed (owner chose local-only); Cellpose downloads disable TLS checks process-wide and load a pickled size model without integrity check; round 5's `--extra-index-url` made uv take outdated requests 2.28.1 / urllib3 1.26.13 / certifi 2022 from the PyTorch mirror; usernames/emails/local paths in git history (committed .pyc, author emails); any Pillow format decoded despite PNG/JPEG UI; Streamlit defaults (200 MB uploads, full tracebacks); ResNet confusion matrix generated but never shown; no .gitattributes / .gitignore gaps; unpinned uv; server paths in page warnings | 7: `--find-links` (uv and pip verified: latest requests/urllib3/certifi, CPU torch, 0 nvidia); Cellpose TLS restored + SHA-256 pinned (hashes re-verified over verified TLS) + CI cache key tied to hashes; uploads PNG/JPEG only; `.streamlit/config.toml`; ResNet matrix on page 3; `.gitattributes`, `.gitignore`; uv pinned in CI; file names only in errors. Owner decisions: push, history rewrite for privacy, Git LFS, SHA-pinning actions | 121 → 124 |
| 9 | Documentation ↔ code consistency | 14 verified: README said plain `python train_medical.py` produces the shipped model (it trains the 96.0% default and overwrites the shipped 96.7% files); "CPU wheel index" wording stale after the `--find-links` switch; torch versions repeated in README; notebook described as covering every module, with stale method lists and summary table; Project Structure missing 8 files; `src/__init__` missing `image_io`; requirements-ml understated sklearn need; CLAUDE.md CI description and `\?\` pointer inaccurate; pytest marker text wrong; ResNet gain 1.2 → 1.3 points; mlp default epochs wrong in `parse_args`; "optional torch" docstrings; stale comments/captions; scripts without `--help` text | All 14 | 124 → 124 |
| 10 | Full fresh sweep, regressions from rounds 1–9 | 8 verified: negative label ids gave F1 = 3.0 and counted the background (regression from round 7's id compaction); round 7 memory fix had no test; `train_medical` checked the weights folder but not the figure folder; mismatched checkpoints (RuntimeError/KeyError/AttributeError) took down page 3; 5000×1 uploads crashed every page's preview; two unmarked tests ran Cellpose; `tomllib` test failed on Python 3.10; nits (color docstring, unfiltered GT panel, README `--out` example and page-3 wording, duplicated results-path logic) | All 8; compaction test with a tracemalloc bound (3 mutants re-run, all caught) | 124 → 129 |
| 11 | Fresh-clone README walkthrough, Windows→Linux, multi-user state | 3 verified (no high): benchmark script overwrote the shipped figure by default, as a 4-panel image without Cellpose; README setup lacked clone step, Linux/macOS/cmd activation, a test install line for Python 3.13+ and the notebook dependency; two test files only passed from the repo root. Clean: case-sensitive paths, LF endings, file modes, shared cached models under 8 concurrent threads | All 3 | 129 → 130 |

### Upgrades after the loop (2026-10-13)

Real-data benchmark (BBBC039), U-Net, Grad-CAM, calibration. Found and fixed
while building them: weighted 2-D cross-entropy has no deterministic CUDA
kernel (hand-written equivalent, verified equal); `fit_temperature` did not
converge (2.43 vs optimum 2.97); U-Net decoding one pixel too large (found on
the first test evaluation, fix chosen on validation, disclosed in the README).
Grad-CAM verified against pytorch-grad-cam (correlation 1.0000). Tests 130 → 144.

**Loop summary (2026-10-13):** 11 rounds, 90 verified findings fixed, tests 29 → 130, every README number re-measured from committed scripts. Open owner decisions: push to GitHub + reboot Streamlit Cloud; LICENSE; source of the brightfield demo photo; rewriting git history to remove local paths/emails in old .pyc files and commit metadata; Git LFS for model weights.
