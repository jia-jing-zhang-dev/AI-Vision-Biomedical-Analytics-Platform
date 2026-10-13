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
    st.write("- Supports real-time image uploads for 10-class classification (86.9% test accuracy).")

    st.image(
        "public/fashion-demo.png",
        caption="FashionMNIST Classification Demo",
        width="stretch"
    )

with col2:
    st.subheader("🔬 Module 2: Biomedical Image Segmentation")
    st.write(
        "- Six classical computer vision methods, including Otsu thresholding, "
        "top-hat background correction and a watershed that splits touching cells."
    )
    st.write(
        "- Benchmarked with **Dice / IoU** and **instance F1**: the watershed raises "
        "instance F1 from 0.60 (Otsu) to 0.80 on images of touching cells; "
        "Cellpose, the standard deep-learning tool, is included as a reference (0.90)."
    )

    st.image(
        "public/cell-segmentation-benchmark.png",
        caption="Synthetic Cell Benchmark: Ground Truth vs. Otsu vs. Watershed",
        width="stretch"
    )

    st.image(
        "public/brightfield-demo.png",
        caption="Real Blood Smear: Automatic Inversion + Watershed",
        width="stretch"
    )

with col3:
    st.subheader("🩸 Module 3: Blood Cell Classification")
    st.write("- A PyTorch CNN that classifies 8 blood-cell types (BloodMNIST).")
    st.write(
        "- **96.7% test accuracy**, up from 72.9% for a fully connected baseline; "
        "hyperparameters tuned with Optuna on the validation split."
    )
    st.write(
        "- Compared with an ImageNet-pretrained ResNet-18 (97.9%, 20× larger) and the "
        "published MedMNIST v2 benchmarks."
    )

    st.image(
        "public/bloodmnist-confusion.png",
        caption="BloodMNIST Test-Set Confusion Matrix",
        width="stretch"
    )