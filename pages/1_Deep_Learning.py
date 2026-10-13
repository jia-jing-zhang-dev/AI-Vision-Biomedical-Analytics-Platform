import streamlit as st
import torch

from src.image_io import UploadError, load_upload
from src.vision_cnn import FASHION_MNIST_CLASSES, build_model, prepare_upload

st.set_page_config(page_title="Vision Analytics Framework", layout="centered")

st.title("FashionMNIST Image Classification System")
st.write("An End-to-End Visual Analysis Framework Built with PyTorch")


@st.cache_resource
def get_model():
    model, device = build_model()
    model.load_state_dict(torch.load("fashion_model.pth", map_location=device, weights_only=True))
    model.eval()
    return model, device


model, device = get_model()

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
        image, _ = load_upload(uploaded_file, mode="RGB")   # 8-bit, upright, at most 1024 px
    except UploadError as err:
        st.error(str(err))
        st.stop()
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Run the PyTorch Model for Classification"):
        with st.spinner("Model inference in progress..."):
            # Grayscale 28x28; inverted only if the background is light
            input_tensor = prepare_upload(image).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_class_idx = output.argmax(1).item()
                predicted_class_name = FASHION_MNIST_CLASSES[predicted_class_idx]
            
            st.success(f"**Prediction Result:** {predicted_class_name}")