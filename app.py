import streamlit as st

st.set_page_config(
    page_title="Vision Analytics Framework",
    layout="wide"
)

st.title("🏥 AI Vision & Biomedical Analytics Platform")

st.write(
    "Welcome to the end-to-end computer vision and biomedical "
    "image analysis framework!"
)

st.caption(
    "The classification models are trained on 28×28 images (FashionMNIST, BloodMNIST)."
)

st.info(
    "👈 Please use the **Navigation** panel on the left "
    "to switch between different modules."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🖼️ Module 1: PyTorch Deep Learning Classification")
    st.write("- A fully connected neural network built with PyTorch.")
    st.write("- Supports real-time image uploads for 10-class classification.")

    st.image(
        "public/fashion-demo.png",
        caption="FashionMNIST Classification Demo",
        use_container_width=True
    )

with col2:
    st.subheader("🔬 Module 2: Biomedical Image Segmentation")
    st.write(
        "- Includes traditional computer vision algorithms "
        "such as Otsu thresholding and Watershed."
    )
    st.write(
        "- Automatically calculates and outputs **Dice / IoU** "
        "performance metrics for biomedical image segmentation."
    )

    st.image(
        "public/segmentation-demo.png",
        caption="Synthetic Biomedical Image Demo",
        use_container_width=True
    )

    st.image(
        "public/Lymphocyte-demo.png",
        caption="Lymphocyte Segmentation Image Demo",
        use_container_width=True
    )

with col3:
    st.subheader("🩸 Module 3: Blood Cell Classification")
    st.write("- A PyTorch CNN that classifies 8 blood-cell types (BloodMNIST).")
    st.write(
        "- **96.6% test accuracy**, up from 72.9% for a fully connected baseline; "
        "hyperparameters tuned with Optuna on the validation split."
    )

    st.image(
        "public/bloodmnist-confusion.png",
        caption="BloodMNIST Test-Set Confusion Matrix",
        use_container_width=True
    )