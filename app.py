import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.vision_cnn import build_model

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.set_page_config(page_title="Vision Analytics Framework", layout="centered")

st.title("FashionMNIST Image Classification System")
st.write("An End-to-End Visual Analysis Framework Built with PyTorch")


model, device = build_model()

model.load_state_dict(torch.load("fashion_model.pth", map_location=device, weights_only=True))
model.eval() 

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),                 
    transforms.ToTensor()                       
])

uploaded_file = st.file_uploader("Please upload an image of clothing (jpg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Run the PyTorch Model for Classification"):
        with st.spinner("Model inference in progress..."):
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            input_tensor = 1.0 - input_tensor
            
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class_idx = output.argmax(1).item()
                predicted_class_name = FASHION_MNIST_CLASSES[predicted_class_idx]
            
            st.success(f"**Prediction Result：** {predicted_class_name}")