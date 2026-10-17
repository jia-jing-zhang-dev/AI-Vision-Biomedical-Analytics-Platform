"""224-pixel blood-cell models, foundation-model probes and the cross-source
evaluation (train_medical --model resnet18_224, probe_foundation.py,
eval_external.py). Run with:
    pytest tests/
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_resnet18_224_reads_uploads_at_224_pixels():
    """The 28-pixel models reduce uploads to 28x28 first; the 224 model must
    keep the detail and reduce straight to 224x224."""
    from src.medical_cnn import input_transform
    upload = Image.new("RGB", (368, 368), (200, 120, 160))
    assert input_transform("resnet18_224", from_upload=True)(upload).shape == (3, 224, 224)
    assert input_transform("cnn", from_upload=True)(upload).shape == (3, 28, 28)


def test_train_medical_knows_the_224_model():
    import train_medical
    args = train_medical.parse_args(["--model", "resnet18_224"])
    assert args.epochs == 10
    paths = train_medical.output_paths("resnet18_224")
    assert paths["weights"] == "biomedical_resnet18_224.pth"
    assert paths["figure"] == "public/bloodmnist-confusion-resnet18-224.png"


def test_probe_head_folds_standardization_exactly():
    """probe_foundation saves scaler + logistic regression as one linear layer;
    it must give the same probabilities as the fitted sklearn pipeline."""
    from probe_foundation import probe
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 3, (8, 16))
    def split(n):
        y = rng.integers(0, 8, n)
        return centres[y] * rng.uniform(0.5, 2, 16) + rng.normal(5, 1, (n, 16)), y
    feats, labels = {}, {}
    for s, n in (("train", 400), ("val", 100), ("test", 100)):
        feats[s], labels[s] = split(n)
    _, _, _, proba, head = probe(feats, labels)
    logits = feats["test"] @ head["weight"].T + head["bias"]
    folded = np.exp(logits - logits.max(1, keepdims=True))
    folded /= folded.sum(1, keepdims=True)
    np.testing.assert_allclose(folded, proba, atol=1e-4)


def test_external_class_mapping_matches_bloodmnist():
    from eval_external import KU_TO_BLOODMNIST
    from src.medical_cnn import CLASS_NAMES
    assert set(KU_TO_BLOODMNIST.values()) <= set(CLASS_NAMES)
    assert KU_TO_BLOODMNIST["band_neutrophil"] == "neutrophil"           # Acevedo counts bands as neutrophils
    assert KU_TO_BLOODMNIST["myelocyte"] == "immature granulocyte"
    for excluded in ("blast", "reactive_lymphocyte", "giant_platelet", "platelet_cluster"):
        assert excluded not in KU_TO_BLOODMNIST


def test_external_metrics_bootstrap_by_patient():
    """Macro-F1 averages the present classes; the CI resamples whole patients,
    so a perfect predictor has a degenerate interval and a poor one a wide one."""
    from eval_external import metrics
    labels = np.array([0, 0, 1, 1, 6, 6, 6, 6])
    patients = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    perfect = metrics(labels.copy(), labels, patients, n_boot=200)
    assert perfect["accuracy"] == 1.0 and perfect["macro_f1"] == 1.0 and perfect["accuracy_ci"] == [1.0, 1.0]
    wrong = labels.copy(); wrong[labels == 6] = 0                         # every neutrophil missed
    m = metrics(wrong, labels, patients, n_boot=200)
    assert m["recall"]["neutrophil"] == 0.0 and m["accuracy"] == 0.5
    assert m["accuracy_ci"][0] < m["accuracy"] < m["accuracy_ci"][1]


@pytest.mark.parametrize("argv", [["--data-dir", "no_such_dir", "--split", "test"],
                                  ["--data-dir", ".", "--split", "train"]])
def test_eval_external_rejects_bad_arguments(argv):
    import eval_external
    with pytest.raises(SystemExit) as exc:
        eval_external.main(argv)
    assert exc.value.code == 2


def test_probe_foundation_needs_the_224_file(tmp_path):
    import probe_foundation
    with pytest.raises(SystemExit) as exc:
        probe_foundation.main(["--data-dir", str(tmp_path)])
    assert exc.value.code == 2


def test_dinobloom_checkpoint_loads_offline_and_classifies_samples(monkeypatch):
    """The app model loads from the repository alone (no Hugging Face access)
    and gets the bundled 224-px samples right."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from src.medical_cnn import CLASS_NAMES, load_model, predict_proba
    model = load_model(str(ROOT / "biomedical_dinobloom_s.pth"))
    assert model.arch == "dinobloom_s"
    samples = sorted((ROOT / "public" / "blood_samples").glob("*.png"))
    hits = sum(CLASS_NAMES[int(predict_proba(model, Image.open(p)).argmax())] ==
               p.stem.split("_", 1)[1].replace("_", " ") for p in samples)
    assert hits >= 7, hits


def test_vit_grad_cam_matches_pytorch_grad_cam():
    """Hook-free Grad-CAM for the ViT equals pytorch-grad-cam on blocks[-1].norm1."""
    pytest.importorskip("pytorch_grad_cam")
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.explain import _features_and_head, grad_cam
    from src.medical_cnn import input_transform, load_model
    model = load_model(str(ROOT / "biomedical_dinobloom_s.pth"))
    img = Image.open(ROOT / "public" / "blood_samples" / "5_monocyte.png")
    x = input_transform("dinobloom_s", from_upload=True)(img).unsqueeze(0)
    acts, head = _features_and_head(model, x)
    assert float((head(acts) - model(x)).abs().max()) < 1e-4          # the split is the model
    mine, target = grad_cam(model, img)
    reshape = lambda t: t[:, 1:, :].reshape(t.size(0), 16, 16, t.size(2)).permute(0, 3, 1, 2)
    ref = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=reshape)(
        input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
    assert np.corrcoef(mine.ravel(), ref.ravel())[0, 1] > 0.999


def test_external_results_match_the_page_and_readme():
    import json
    ext = json.loads((ROOT / "external_results.json").read_text())["test_reframed"]
    assert round(ext["DinoBloom-S probe"]["accuracy"], 3) == 0.938
    assert ext["DinoBloom-S probe"]["accuracy"] > ext["ResNet-18 (224 px)"]["accuracy"] + 0.1
