"""
Cross-source evaluation of the blood-cell models on KU-Optofil PBC
(Sysmex DI-60, May-Grunwald-Giemsa; Zenodo 10.5281/zenodo.17333317, CC BY 4.0),
a different laboratory and scanner from BloodMNIST (Acevedo et al.: CellaVision DM96,
also May-Grunwald-Giemsa). No model is
trained on it: this measures transfer to a new source.

    python eval_external.py --data-dir DIR --split validation   # choose models / thresholds
    python eval_external.py --data-dir DIR --split test         # report once
    python eval_external.py --data-dir DIR --split test --reframe   # zoomed-out cells re-framed first
    python eval_external.py --data-dir DIR --split test --reframe --perturb strong   # + colour changes

DIR holds dataset.zip and metadata_with_patient_level_splits.csv. Classes
are mapped to BloodMNIST's (band and segmented neutrophils -> neutrophil;
myelocytes and metamyelocytes -> immature granulocyte); giant platelets,
platelet clusters, blasts and reactive lymphocytes have no BloodMNIST
counterpart and are left out. Models still predict all 8 classes.
Confidence intervals resample patients (images of one patient are correlated).
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.medical_cnn import CLASS_NAMES, input_transform, load_model

KU_TO_BLOODMNIST = {
    "segmented_neutrophil": "neutrophil", "band_neutrophil": "neutrophil",
    "metamyelocyte": "immature granulocyte", "myelocyte": "immature granulocyte",
    "lymphocyte": "lymphocyte", "monocyte": "monocyte", "eosinophil": "eosinophil",
    "basophil": "basophil", "erythroblast": "erythroblast",
}
RESULTS = "external_results.json"


def load_split(data_dir: Path, split: str):
    """(PIL images, BloodMNIST labels, patient ids) of the mapped cells of one split."""
    import pandas as pd
    meta = pd.read_csv(data_dir / "metadata_with_patient_level_splits.csv")
    meta = meta[(meta["split"] == split) & meta["cell_type"].isin(list(KU_TO_BLOODMNIST))]
    images = []
    with zipfile.ZipFile(data_dir / "dataset.zip") as zf:
        index = {Path(n).name: n for n in zf.namelist() if n.lower().endswith((".jpg", ".png"))}
        for name in meta["image_name"]:
            with zf.open(index[name]) as f:
                images.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
    labels = np.array([CLASS_NAMES.index(KU_TO_BLOODMNIST[c]) for c in meta["cell_type"]])
    return images, labels, meta["patient_id"].to_numpy()


@torch.no_grad()
def probs_trained(weights: str, images, device: str) -> np.ndarray:
    """Class probabilities of a train_medical.py checkpoint (with its calibration)."""
    model = load_model(weights, device=device)
    tfm = input_transform(model.arch, from_upload=True)
    out = []
    for i in range(0, len(images), 128):
        x = torch.stack([tfm(im) for im in images[i:i + 128]]).to(device)
        out.append(torch.softmax(model(x) / model.temperature, 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def probs_probe(name: str, cache_dir: Path, images, device: str) -> np.ndarray:
    """Class probabilities of a frozen foundation model + its linear probe."""
    from probe_foundation import build_backbone, embed
    model, mean, std = build_backbone(name, device)
    arr = np.stack([np.asarray(im.resize((224, 224), Image.Resampling.BOX)) for im in images])
    feats = embed(model, mean, std, arr, device)
    head = np.load(cache_dir / f"{name}_head.npz")
    logits = feats @ head["weight"].T + head["bias"]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def metrics(pred, labels, patients, n_boot: int = 1000, seed: int = 0):
    """Accuracy and macro-F1 over the present classes, with patient-bootstrap 95% CIs."""
    from sklearn.metrics import f1_score
    present = np.unique(labels)

    def score(idx):
        return ((pred[idx] == labels[idx]).mean(),
                f1_score(labels[idx], pred[idx], labels=present, average="macro", zero_division=0))

    acc, f1 = score(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    boot = np.array([score(np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]))
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    recall = {CLASS_NAMES[c]: float((pred[labels == c] == c).mean()) for c in present}
    return {"accuracy": float(acc), "accuracy_ci": [float(lo[0]), float(hi[0])],
            "macro_f1": float(f1), "macro_f1_ci": [float(lo[1]), float(hi[1])], "recall": recall}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-source evaluation on KU-Optofil PBC.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--resnet224", default="biomedical_resnet18_224.pth")
    parser.add_argument("--cache-dir", default="probe_cache", help="probe heads from probe_foundation.py")
    parser.add_argument("--perturb", choices=["mild", "strong"], default=None,
                        help="apply src.color_shift.perturb (seeded per image) after re-framing")
    parser.add_argument("--reframe", action="store_true",
                        help="re-frame zoomed-out cells to BloodMNIST's scale (src.blood_cells.reframe_single_cell)")
    args = parser.parse_args(argv)
    data_dir, cache = Path(args.data_dir), Path(args.cache_dir)
    for f in ("dataset.zip", "metadata_with_patient_level_splits.csv"):
        if not (data_dir / f).exists():
            parser.error(f"{data_dir / f} not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels, patients = load_split(data_dir, args.split)
    if args.reframe:
        from src.blood_cells import reframe_single_cell
        images = [reframe_single_cell(im) for im in images]
    if args.perturb:
        from src.color_shift import perturb
        images = [perturb(im, args.perturb, i) for i, im in enumerate(images)]
    key = args.split + ("_reframed" if args.reframe else "") + (f"_{args.perturb}" if args.perturb else "")
    print(f"{args.split}: {len(images)} cells from {len(np.unique(patients))} patients")
    candidates = {
        "DinoBloom-S (app model)": lambda: probs_trained("biomedical_dinobloom_s.pth", images, device),
        "Custom CNN (28 px)": lambda: probs_trained("biomedical_cnn.pth", images, device),
        "ResNet-18 (28 px upsampled)": lambda: probs_trained("biomedical_resnet18.pth", images, device),
        "ResNet-18 (224 px)": lambda: probs_trained(args.resnet224, images, device),
        "DinoBloom-S probe (no jitter)": lambda: probs_probe("dinobloom_s", cache, images, device),
        "DINOv2-S probe": lambda: probs_probe("dinov2_s", cache, images, device),
        "DINOv2-B probe": lambda: probs_probe("dinov2_b", cache, images, device),
    }
    if not Path(args.resnet224).exists():           # not shipped (45 MB); train_medical.py --model resnet18_224
        print(f"skipping ResNet-18 (224 px): {args.resnet224} not found")
        del candidates["ResNet-18 (224 px)"]
    results = json.loads(Path(RESULTS).read_text()) if Path(RESULTS).exists() else {}
    results[key] = {}
    np.savez(cache / f"external_{args.split}_labels.npz", labels=labels, patients=patients)
    for name, run in candidates.items():
        probs = run()
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        np.save(cache / f"external_{key}_{slug}.npy", probs)               # for threshold tuning
        m = metrics(probs.argmax(1), labels, patients)
        results[key][name] = m
        print(f"{name:28s} acc {m['accuracy']:.3f} [{m['accuracy_ci'][0]:.3f}, {m['accuracy_ci'][1]:.3f}]  "
              f"macro-F1 {m['macro_f1']:.3f} [{m['macro_f1_ci'][0]:.3f}, {m['macro_f1_ci'][1]:.3f}]")
    Path(RESULTS).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
