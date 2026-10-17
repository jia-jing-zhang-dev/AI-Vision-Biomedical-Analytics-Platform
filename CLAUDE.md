# CLAUDE.md

Guidance for Claude (and humans) working on this repository.

## Project

Streamlit app + reusable modules for biomedical image analysis, deployed at
<https://visionserve-full-stack-ml-framework.streamlit.app/>.

- `src/medical_cnn.py` — BloodMNIST classifiers: custom CNN (shipped, 96.7%), MLP baseline (72.9%), ImageNet-pretrained ResNet-18 reference (97.9%); `resnet18_224` (99.1%) and `dinobloom_s` (app model, 98.9%; 93.8% on KU-Optofil)
- `src/segmentation.py` — classical segmentation, synthetic cell benchmark, Dice/IoU/instance F1, optional Cellpose reference
- `src/vision_cnn.py` — FashionMNIST CNN `build_cnn` (92.8%, the app's model, `fashion_cnn.pth`) and MLP baseline `build_model` (86.9%, `fashion_model.pth`); `prepare_upload` crops/centres/fills 28 px like FashionMNIST (product photos 12% → 86%, regression test in test_regressions.py)
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
- CI (`.github/workflows/tests.yml`) runs two jobs, **both on Python 3.12**
  (owner's choice, 2026-10-16): dev extras, and the deployment's dependencies
  installed the way Streamlit Cloud does (uv, wheels only) plus
  `requirements-ml.txt` and pytest. Python 3.14, which Streamlit Cloud runs,
  is therefore no longer tested in CI.
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
- Streamlit Cloud docs (Feb 2024 figures) list 690 MB–2.7 GB memory and up to 2
  CPU cores; the app idles at ~765 MB once every page was visited. Wrap
  memory-heavy page work in `src.runtime.heavy_work()` (one at a time per
  process) and call `limit_torch_threads()` on pages that run torch.
- BBBC039 lives in `data/bbbc039/` (gitignored; CC0, ~80 MB from
  bbbc.broadinstitute.org/BBBC039). Use the official split: train the U-Net
  on training, select on validation, evaluate test as rarely as possible
  (`benchmark_bbbc039.py`; it has been evaluated three times, disclosed in
  the README protocol note). Report Dice, F1@0.5 and average F1 (IoU
  0.50–0.90, the paper's metric).
- U-Net classes follow Caicedo et al.: the boundary band includes one pixel
  outside each nucleus, so `classes_to_labels` erodes the foreground by one
  pixel with `border_value=1` (without erosion every nucleus is 1 px too large;
  without border_value=1 nuclei at the image edge lose a row). Images over
  400k pixels run in 384-px tiles (single pass on 1024x1024 cost ~750 MB).
- `load_model` applies `<weights>_calibration.json` (temperature from
  `calibrate_medical.py`); predictions are unchanged, confidences calibrated.
  `fit_temperature` needs the strong-Wolfe line search (it stopped early before).
  `train_medical.py` deletes the calibration JSON when it overwrites weights;
  re-run `calibrate_medical.py` after retraining.
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
5. **Dependencies & deployment** — installs on Python 3.14 from wheels (check
   locally: CI runs 3.12 only); both CI jobs green; versions pinned where
   breakage is likely.
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
Grad-CAM verified against pytorch-grad-cam (correlation 1.0000). Follow-up review: U-Net
memory (tiling), border erosion (third test evaluation, disclosed), calibration claim
reworded with a bootstrap CI (not significant), 9 more tests. Tests 130 → 153.

### Pre-reboot checks (2026-10-15)

| Check | Verified findings | Fixed | Tests |
|---|---|---|---|
| A: deployment rehearsal (clean Python 3.14 venv, uv install of requirements.txt, real server on Windows and Linux/2 CPUs) | App works from a clean clone, no runtime downloads; one visitor peaks ~934 MB and two concurrent 25 MP uploads ~1.16 GB; Grad-CAM checkbox reset when switching image source | `heavy_work()` lock around uploads/segmentation/inference, torch limited to 2 threads, lazy pyplot import, checkbox always drawn | 153 → 155 |
| B: full code review (lock, cross-module interactions) | No high: lock is never nested and released on Streamlit stop/rerun; tiling vs erosion verified on 1040×1392 mosaics. Found: stale calibration JSON applied after retraining; malformed/zero/negative temperature crashed page 3 or flipped predictions; `train_unet --crop/--boundary-weight` unvalidated; README structure and `src/__init__` missing new modules; ambiguous README summary row | All: `read_temperature` validation, `train_medical` removes the stale calibration file, CLI checks, docs | 155 → 159 |
| C: scientific rigor (re-ran every BBBC039 method incl. Cellpose, paper's evaluation code, McNemar on BloodMNIST, citations) | "U-Net best / finds the most nuclei" unsupported (its lead is one image, IXMtest_F12_s8, where Cellpose scores 0; Cellpose better on 35/50, sign test p=0.001); ground truth filtered at 20 px unlike the paper (156 of 5,720 nuclei; all scores −0.011 to −0.014); Optuna significance used the wrong test; "evaluates test once" contradicted the protocol note; adaptive "best classical" only by average F1/Dice; Grad-CAM "exactly", calibration "calibrated" and CNN-gain attribution overclaimed; U-Net, Acevedo (BloodMNIST source), FashionMNIST venue, Cellpose3 uncited | All: `evaluate(gt_min_size=)`, benchmark reports the paper protocol and per-image scores (old numbers reproduce exactly), README/app wording, McNemar numbers (p=0.013, p=2.4e-6) re-measured, credits. Open: possible overlap of Cellpose's training data with BBBC039 (unconfirmed) | 159 → 160 |

### Live-demo feedback (2026-10-15, owner's own uploads)

| Upload | What happened | Diagnosis | Status |
|---|---|---|---|
| FashionMNIST: product photo of a low ankle boot | Predicted "Sandal" | Whole photo squeezed to 28x28: item small, distorted, on grey; FashionMNIST items are cropped, centred and fill 28 px | Fixed: `prepare_upload` crops/centres/fills; product-photo-like test uploads 12% → 86%; page shows the 28x28 input and top-3. Second cause found after deploying: the boot faced right, every FashionMNIST shoe faces left (mirrored boots 0%); left-heavy items are now mirrored (rule checked on train, test all-mirrored 63.5% → 86.5%, unmirrored −0.4) |
| Segmentation: SEM image of a neuron | U-Net splits the neuron into many pieces | Out of domain (trained on fluorescent nuclei); the page caption already warns | No change |
| Blood cells: probability chart | ResNet-18 chart x-axis ran from −2 to 1 | Streamlit bar chart auto-scales its axis | Fixed: Altair chart with a fixed 0–1 axis |
| Blood cells: smear with many cells | Several neutrophils → "immature granulocyte" | Models expect one centred cell per image (BloodMNIST crops) | Fixed: `src/blood_cells.py` finds stained cells (B−G), one crop size per image (median stained size / 0.60, from validation), page shows numbered overview + per-cell predictions. Test mosaics (900 cells): 99.0% of leukocytes/erythroblasts found, crops classified 86.4% CNN / 89.7% ResNet vs 95.7/97.3% on original crops; platelets mostly skipped; TTA did not help (validation). Smear mode only for ≥2 stained cells (a size rule flagged 18% of single crops) |

Follow-up (same day): the boot still failed after both preprocessing fixes (very low, long shape; mirror rule saw it as right-heavy). Page 1 now uses a CNN trained with flip/shift/scale augmentation (`train.py`, epoch chosen on a 5,000-image validation part of the training set, test once): test 92.8% vs 86.9%, mirrored test 92.5% vs 63.5%, photo-like uploads 90.5% vs 85.8%, McNemar p < 1e-70; 140 k vs 670 k parameters. The MLP stays as the comparison baseline.

Smear report (2026-10-16): owner's real neutrophil smear (other source) — all 9 cells found, but CNN 3/9 and ResNet-18 6/9 neutrophil (stain/scanner shift; anecdotal, n=9). Added `consensus` + `differential_count` (`src/blood_cells.py`): counted only when both models agree with >= REVIEW_CONFIDENCE 0.85 each, chosen on validation mosaics (lowest t with >= 98% accepted accuracy); test mosaics once: 77.0% counted, 98.3% correct, 103/113 errors flagged. Tests 170 -> 172.

Lessons from this feedback: preprocessing for uploads must reproduce how the
training images were made (FashionMNIST: crop, centre, fill; BloodMNIST: one
centred cell). Check every page with a realistic upload of the kind a visitor
would try (product photo, whole smear), not only with dataset samples. Tests
166 (161 → 166: smear unit tests, smear page test).

### Plan A: 224 px, foundation models, second scanner (2026-10-17)

- Data (not in git): MedMNIST+ `bloodmnist_224.npz` (Zenodo 10519652, 1.5 GB, MD5 b718ff68...) and
  KU-Optofil PBC (Zenodo 17333317, CC BY 4.0, 565 MB; Sysmex DI-60, MGG; published after DinoBloom).
  Kept in a short temp path (long-path issues).
- ResNet-18 at native 224 px: test 99.12% (MedMNIST+ reports 98.94); vs 97.92% at 28->224, p=2e-7.
  Weights not shipped; results JSON is.
- `probe_foundation.py`: frozen DinoBloom-S / DINOv2-S/B + logistic head (C on validation):
  98.95 / 97.75 / 98.07%. DinoBloom-S vs ResNet-224 p=0.39; vs DINOv2 p<=2e-4.
- `eval_external.py` on KU-Optofil (classes mapped, 4 excluded; patient-level split; validation to choose,
  test once; patient bootstrap CIs). First run: trained models 2-5% (cells half the size; ResNet-224 said
  "erythroblast" for 96%). Fix chosen on validation: `reframe_single_cell` (stained/frame < 0.45 ->
  crop at 0.661). RBC-size magnification estimate tried and dropped. Test (re-framed): DinoBloom-S 93.8%
  [91.7, 95.4] macro-F1 0.77; ResNet-224 76.2%; DINOv2-B 86.0%; CNN 65.0%.
- App: page 3 now DinoBloom-S (exported head+backbone, 86.6 MB, `timm` in requirements, loads offline)
  + custom CNN baseline; ResNet-18 dropped from the page. Review rule: DinoBloom-S confidence < 0.70
  (chosen on KU validation; KU test 92.6% accepted at 97.1%). Smear mosaics of 224 test cells: 97.9%
  found, 98.9% correct. Bundled samples replaced by 224-px originals of the same test images (the 28-px
  upscales gave DinoBloom 2/8). ViT Grad-CAM split at blocks[-1].norm1 (pytorch-grad-cam corr 1.000;
  splitting before norm1 gave 0.43-0.96). Memory on Python 3.14 after all pages: 607 MB.
- Owner decision: weights in git (option A). Tests 172 -> 184.

**Loop summary (2026-10-13):** 11 rounds, 90 verified findings fixed, tests 29 → 130, every README number re-measured from committed scripts. Open owner decisions: push to GitHub + reboot Streamlit Cloud; LICENSE; source of the brightfield demo photo; rewriting git history to remove local paths/emails in old .pyc files and commit metadata; Git LFS for model weights.
