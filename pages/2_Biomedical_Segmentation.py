import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from src.segmentation import make_synthetic_image, run_all, evaluate

st.title("🔬 Biomedical Image & Cell Segmentation System")
st.write(
    "Supports both **Benchmark Evaluation Mode** and **Custom Medical Image Upload Mode**, "
    "using multiple algorithms to extract cell/lesion boundaries and calculate Dice / IoU metrics."
)

# Mode selection
mode = st.radio(
    "Select Operation Mode",
    [
        "1. Generate Synthetic Biomedical Image (Benchmark Evaluation)",
        "2. Upload Custom Medical Image (Interactive Inference)"
    ]
)

if mode == "1. Generate Synthetic Biomedical Image (Benchmark Evaluation)":
    if st.button("Run Biomedical Image Segmentation Pipeline"):
        img, gt_mask = make_synthetic_image()
        results = run_all(img)
        scores = evaluate(gt_mask, results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Medical Scan")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Expert-Annotated Ground Truth")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(gt_mask, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            
        st.subheader("📊 Algorithm Performance Evaluation (Dice / IoU)")
        
        score_data = []
        for item in scores:
            if isinstance(item, dict):
                score_data.append({
                    "Method": item.get("method", "Alg"),
                    "Dice": round(item.get("dice", 0.0), 3),
                    "IoU": round(item.get("iou", 0.0), 3)
                })
            else:
                score_data.append({
                    "Method": item[0],
                    "Dice": round(item[1], 3),
                    "IoU": round(item[2], 3)
                })
                
        st.table(score_data)
        
        st.subheader("👁️ Segmentation Mask Visualization")
        cols = st.columns(len(results))
        
        for i, (name, mask) in enumerate(results.items()):
            with cols[i]:
                st.text(name)
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                ax.imshow(mask, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)

else:
    st.subheader("📤 Upload Your Own Medical Image")
    
    uploaded_file = st.file_uploader(
        "Upload a cell, X-ray, or pathology image...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Read and convert to grayscale

        pil_img = Image.open(uploaded_file).convert("L")
        img = np.array(pil_img)
        
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
            
        st.image(
            pil_img,
            caption="Uploaded Medical Image",
            width=350
        )
        
        if st.button("Run Multi-Algorithm Segmentation"):
            with st.spinner("Running classical computer vision algorithms..."):
                results = run_all(img)
                
            st.subheader("👁️ Segmentation Results for Uploaded Image")
            
            cols = st.columns(len(results))
            
            for i, (name, mask) in enumerate(results.items()):
                with cols[i]:
                    st.text(name)
                    fig, ax = plt.subplots(figsize=(2.5, 2.5))
                    ax.imshow(mask, cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)