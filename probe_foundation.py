"""
Linear probing of pretrained foundation models on BloodMNIST (224x224).

    python probe_foundation.py --data-dir DIR          # DIR holds bloodmnist_224.npz (MedMNIST+)
    python probe_foundation.py --data-dir DIR --models dinobloom_s
    python probe_foundation.py --data-dir DIR --models dinobloom_s --export biomedical_dinobloom_s.pth

Each backbone is frozen; its image embedding (CLS token) feeds a logistic
regression. The regularization strength is chosen on the validation split
and the test split is evaluated once. Embeddings are cached in --cache-dir,
with each fitted probe folded into one linear layer (<name>_head.npz),
results go to foundation_probe_results.json and test-set probabilities to
--cache-dir for paired comparisons.

Backbones (weights from the Hugging Face Hub via timm, Apache-2.0):
- dinobloom_s: DinoBloom-S, a DINOv2 ViT-S/14 trained on 380k blood and bone
  marrow cell images; Acevedo et al. (the source of BloodMNIST) was held out
  of its training (Koch et al., MICCAI 2024). Community timm conversion
  1aurent/vit_small_patch14_224.dinobloom of the official weights.
- dinov2_s / dinov2_b: general-purpose DINOv2 ViT-S/14 and ViT-B/14
  (Oquab et al., 2023), run at 224x224.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

BACKBONES = {
    "dinobloom_s": ("hf-hub:1aurent/vit_small_patch14_224.dinobloom", {}),
    "dinov2_s": ("vit_small_patch14_dinov2.lvd142m", {"img_size": 224}),
    "dinov2_b": ("vit_base_patch14_dinov2.lvd142m", {"img_size": 224}),
}
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
RESULTS = "foundation_probe_results.json"


def build_backbone(name: str, device: str):
    import timm
    model_id, kwargs = BACKBONES[name]
    model = timm.create_model(model_id, pretrained=True, num_classes=0, **kwargs).to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    return model, torch.tensor(cfg["mean"]).view(1, 3, 1, 1), torch.tensor(cfg["std"]).view(1, 3, 1, 1)


@torch.no_grad()
def embed(model, mean, std, images: np.ndarray, device: str, batch: int = 128) -> np.ndarray:
    """CLS embeddings of uint8 HxWx3 images (already 224x224)."""
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2).float().div(255)
        x = ((x - mean) / std).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def probe(features: dict, labels: dict):
    """Standardized features -> logistic regression; C chosen on validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features["train"])
    tr, va, te = (scaler.transform(features[s]) for s in ("train", "val", "test"))
    val_acc = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000).fit(tr, labels["train"])
        val_acc[c] = float((clf.predict(va) == labels["val"]).mean())
    best_c = max(val_acc, key=val_acc.get)
    clf = LogisticRegression(C=best_c, max_iter=3000).fit(tr, labels["train"])
    proba = clf.predict_proba(te)                                  # test split: once
    # Standardization + logistic regression folded into one linear layer on raw embeddings
    weight = clf.coef_ / scaler.scale_
    bias = clf.intercept_ - (clf.coef_ * scaler.mean_ / scaler.scale_).sum(1)
    head = {"weight": weight.astype(np.float32), "bias": bias.astype(np.float32)}
    return best_c, val_acc, float((proba.argmax(1) == labels["test"]).mean()), proba, head


def export_dinobloom(cache: Path, out: str, result: dict):
    """One timm model = frozen DinoBloom-S backbone + the fitted linear probe,
    saved like train_medical.py checkpoints (loadable offline by load_model)."""
    from src.medical_cnn import build_dinobloom_s
    backbone, _, _ = build_backbone("dinobloom_s", "cpu")
    model = build_dinobloom_s()
    missing, unexpected = model.load_state_dict(backbone.state_dict(), strict=False)
    assert set(missing) == {"head.weight", "head.bias"} and not unexpected, (missing, unexpected)
    head = np.load(cache / "dinobloom_s_head.npz")
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(head["weight"]))
        model.head.bias.copy_(torch.from_numpy(head["bias"]))
    params = {"arch": "dinobloom_s", "backbone": BACKBONES["dinobloom_s"][0], "probe_C": result["C"]}
    torch.save({"params": params, "state_dict": model.state_dict()}, out)
    Path(out).with_name(Path(out).stem + "_results.json").write_text(json.dumps(
        {**params, "val_accuracy": result["val_accuracy"], "test_accuracy": result["test_accuracy"]}, indent=2))
    print(f"Saved {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Linear probes of foundation models on BloodMNIST 224.")
    parser.add_argument("--data-dir", required=True, help="folder with bloodmnist_224.npz")
    parser.add_argument("--models", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    parser.add_argument("--cache-dir", default="probe_cache", help="embeddings and test probabilities")
    parser.add_argument("--export", metavar="WEIGHTS", default=None,
                        help="save DinoBloom-S + its probe as one checkpoint for the app (needs dinobloom_s)")
    args = parser.parse_args(argv)
    if args.export and "dinobloom_s" not in args.models:
        parser.error("--export needs dinobloom_s among --models")
    npz = Path(args.data_dir) / "bloodmnist_224.npz"
    if not npz.exists():
        parser.error(f"{npz} not found (MedMNIST+ BloodMNIST, size 224)")
    cache = Path(args.cache_dir)
    cache.mkdir(exist_ok=True)

    from src.medical_cnn import get_device, set_deterministic
    set_deterministic(0)
    torch.use_deterministic_algorithms(False)       # inference only; some ViT kernels lack a deterministic path
    device = get_device()
    data = np.load(npz)
    labels = {s: data[f"{s}_labels"][:, 0] for s in ("train", "val", "test")}

    results = json.loads(Path(RESULTS).read_text()) if Path(RESULTS).exists() else {}
    for name in args.models:
        feats, backbone = {}, None                  # backbone loaded only if an embedding is missing
        for split in ("train", "val", "test"):
            path = cache / f"{name}_{split}.npy"
            if not path.exists():
                backbone = backbone or build_backbone(name, device)
                np.save(path, embed(*backbone, data[f"{split}_images"], device))
            feats[split] = np.load(path)
        del backbone
        best_c, val_acc, test_acc, proba, head = probe(feats, labels)
        np.save(cache / f"{name}_test_proba.npy", proba)
        np.savez(cache / f"{name}_head.npz", **head)             # logits = embedding @ weight.T + bias
        results[name] = {"backbone": BACKBONES[name][0], "feature_dim": int(feats["train"].shape[1]),
                         "C": best_c, "val_accuracy": val_acc[best_c], "test_accuracy": test_acc}
        print(f"{name}: dim {feats['train'].shape[1]}, C={best_c}, val {val_acc[best_c]:.4f}, test {test_acc:.4f}")
        Path(RESULTS).write_text(json.dumps(results, indent=2))
    if args.export:
        export_dinobloom(cache, args.export, results["dinobloom_s"])


if __name__ == "__main__":
    main()
