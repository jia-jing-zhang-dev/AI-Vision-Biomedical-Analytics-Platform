import json
from pathlib import Path

import streamlit as st
import torch

from src.image_io import UploadError, load_upload
from src.runtime import heavy_work, limit_torch_threads
from src.vision_cnn import FASHION_MNIST_CLASSES, build_cnn, build_model, prepare_upload

st.set_page_config(page_title="Vision Analytics Framework", layout="centered")

st.title("FashionMNIST Image Classification System")
st.write("An End-to-End Visual Analysis Framework Built with PyTorch")


limit_torch_threads()
ROOT = Path(__file__).resolve().parent.parent
# Display name -> (builder, weights written by train.py)
MODELS = {
    "CNN": (build_cnn, ROOT / "fashion_cnn.pth"),
    "Fully connected baseline": (build_model, ROOT / "fashion_model.pth"),
}
MLP_TEST_ACCURACY = 0.869      # train.py --model mlp (README)


@st.cache_resource
def get_models():
    loaded, errors = {}, []
    for name, (builder, path) in MODELS.items():
        if not path.exists():
            errors.append(f"{path.name} not found; run `python train.py`.")
            continue
        try:
            model, device = builder()
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            loaded[name] = model.eval()
        except Exception as err:     # truncated download, git-LFS pointer, other architecture
            errors.append(f"Could not load {path.name} ({err.__class__.__name__}); re-run `python train.py`.")
    return loaded, errors


def cnn_test_accuracy():
    path = ROOT / "fashion_cnn_results.json"
    return json.loads(path.read_text())["test_accuracy"] if path.exists() else None


models, load_errors = get_models()
for message in load_errors:
    st.warning(message)
if not models:
    st.error("No FashionMNIST model weights found. Run `python train.py` first.")
    st.stop()
main_name = next(iter(models))            # the CNN when present

uploaded_file = st.file_uploader(
    "Upload a clothing image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

st.caption(
    "Supported classes: T-shirt/top, Trouser, Pullover, Dress, Coat, "
    "Sandal, Shirt, Sneaker, Bag, Ankle boot."
)

if uploaded_file is not None:
    try:
        with heavy_work():                                    # one decode at a time per process
            image, _ = load_upload(uploaded_file, mode="RGB")   # 8-bit, upright, at most 1024 px
    except UploadError as err:
        st.error(str(err))
        st.stop()
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Run the PyTorch Model for Classification"):
        with st.spinner("Model inference in progress..."):
            # Cropped, centred, 28x28 light-on-black like the FashionMNIST images
            input_tensor = prepare_upload(image)
            with heavy_work(), torch.no_grad():
                probs = {name: torch.softmax(m(input_tensor.to(next(m.parameters()).device)), 1)[0].cpu()
                         for name, m in models.items()}
        top = probs[main_name].topk(3)
        st.success(f"**Prediction Result ({main_name}):** {FASHION_MNIST_CLASSES[top.indices[0].item()]}")
        col_in, col_top = st.columns([1, 2])
        col_in.image(input_tensor[0, 0].numpy(), caption="What the model sees (28×28)",
                     width=140, clamp=True)
        col_top.write("**Top 3**")
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            col_top.progress(p, text=f"{FASHION_MNIST_CLASSES[i]}: {p:.1%}")
        for name, p in probs.items():
            if name != main_name:
                st.caption(f"{name} for comparison: {FASHION_MNIST_CLASSES[int(p.argmax())]} "
                           f"({p.max():.0%})")
        acc = cnn_test_accuracy()
        st.caption(
            (f"The CNN reaches {acc:.1%} test accuracy (fully connected baseline: "
             f"{MLP_TEST_ACCURACY:.1%}). " if acc is not None and main_name == "CNN" else "")
            + "Both models were trained on 28×28 grayscale product images; real photos with "
            "shadows, angles or busy backgrounds can still be misread."
        )
