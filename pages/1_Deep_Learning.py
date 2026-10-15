from pathlib import Path

import streamlit as st
import torch

from src.image_io import UploadError, load_upload
from src.runtime import heavy_work, limit_torch_threads
from src.vision_cnn import FASHION_MNIST_CLASSES, build_model, prepare_upload

st.set_page_config(page_title="Vision Analytics Framework", layout="centered")

st.title("FashionMNIST Image Classification System")
st.write("An End-to-End Visual Analysis Framework Built with PyTorch")


limit_torch_threads()
WEIGHTS = Path(__file__).resolve().parent.parent / "fashion_model.pth"


@st.cache_resource
def get_model():
    model, device = build_model()
    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    return model, device


if not WEIGHTS.exists():
    st.error("fashion_model.pth not found. Run `python train.py` first.")
    st.stop()
try:
    model, device = get_model()
except Exception as err:     # truncated download, git-LFS pointer, other architecture
    st.error(f"Could not load fashion_model.pth ({err.__class__.__name__}). Re-run `python train.py`.")
    st.stop()

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
            # Grayscale 28x28; inverted only if the background is light
            input_tensor = prepare_upload(image).to(device)

            with heavy_work(), torch.no_grad():
                output = model(input_tensor)
                predicted_class_idx = output.argmax(1).item()
                predicted_class_name = FASHION_MNIST_CLASSES[predicted_class_idx]
            
            st.success(f"**Prediction Result:** {predicted_class_name}")