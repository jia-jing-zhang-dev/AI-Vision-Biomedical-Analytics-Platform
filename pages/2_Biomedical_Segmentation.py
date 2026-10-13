import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from src.segmentation import (
    benchmark, evaluate, has_bright_background, label_objects, make_cell_image,
    make_synthetic_image, run_all, cellpose_available,
)

USE_CELLPOSE = cellpose_available()

MAX_UPLOAD_SIDE = 1024

st.title("🔬 Biomedical Image & Cell Segmentation System")
st.write(
    "Supports both **Benchmark Evaluation Mode** and **Custom Medical Image Upload Mode**, "
    "using multiple algorithms to extract cell/lesion boundaries. Methods are scored with "
    "pixel-level **Dice / IoU** and with **instance F1**, which checks whether each cell "
    "was segmented as its own object (a match needs IoU > 0.5 with a true cell)."
)


def color_labels(labels: np.ndarray) -> np.ndarray:
    """Random colour per instance, black background."""
    palette = np.random.default_rng(1).uniform(0.25, 1.0, (labels.max() + 1, 3))
    palette[0] = 0
    return palette[labels]


def show_image(img, title, cmap=None, size=4):
    if size >= 4:
        st.subheader(title)
    else:
        st.text(title)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)


def show_masks(results):
    st.subheader("👁️ Segmentation Results (one colour per detected object)")
    cols = st.columns(3)
    for i, (name, mask) in enumerate(results.items()):
        with cols[i % 3]:
            show_image(color_labels(label_objects(mask)), name, size=2.5)


# Mode selection
mode = st.radio(
    "Select Operation Mode",
    [
        "1. Generate Synthetic Biomedical Image (Benchmark Evaluation)",
        "2. Upload Custom Medical Image (Interactive Inference)"
    ]
)

if mode == "1. Generate Synthetic Biomedical Image (Benchmark Evaluation)":
    image_type = st.selectbox(
        "Benchmark image",
        ["Realistic cells: touching nuclei, uneven illumination, blur, noise",
         "Simple shapes: three large objects on a mild gradient"],
    )
    realistic = image_type.startswith("Realistic")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1) if realistic else 0

    if st.button("Run Biomedical Image Segmentation Pipeline"):
        if realistic:
            img, gt = make_cell_image(seed=int(seed))
        else:
            img, gt = make_synthetic_image()
        with st.spinner("Running every method" + (" (Cellpose takes a few seconds)" if USE_CELLPOSE else "") + "..."):
            results = run_all(img, include_cellpose=USE_CELLPOSE)
        scores = evaluate(gt, results)

        col1, col2 = st.columns(2)
        with col1:
            show_image(img, "Original Medical Scan", cmap="gray")
        with col2:
            if realistic:
                show_image(color_labels(gt), f"Ground Truth ({gt.max()} cells)")
            else:
                show_image(gt, "Ground Truth", cmap="gray")

        st.subheader("📊 Algorithm Performance Evaluation")
        table = pd.DataFrame(scores).rename(columns={
            "method": "Method", "dice": "Dice", "iou": "IoU", "instance_f1": "Instance F1",
            "cells_true": "Cells (true)", "cells_found": "Objects found",
        })
        if realistic:
            table = table.sort_values("Instance F1", ascending=False)
        st.dataframe(table.round(3), hide_index=True, width="stretch")

        show_masks(results)

    if realistic and st.button("Run 10-Image Benchmark (mean ± std)"):
        with st.spinner("Segmenting 10 images with every method..."):
            summary = benchmark(10)
        st.dataframe(pd.DataFrame([{
            "Method": r["method"],
            "Dice": f"{r['dice']:.3f} ± {r['dice_std']:.3f}",
            "IoU": f"{r['iou']:.3f} ± {r['iou_std']:.3f}",
            "Instance F1": f"{r['instance_f1']:.3f} ± {r['instance_f1_std']:.3f}",
        } for r in summary]), hide_index=True, width="stretch")
        st.caption(
            "Thresholding methods find the foreground well (similar Dice) but merge touching "
            "cells into one object, so their instance F1 is much lower. The watershed splits "
            "them using peaks of the distance transform as seeds. Cellpose is left out here to "
            "keep this fast; the README reports it on 50 images."
        )

else:
    st.subheader("📤 Upload Your Own Medical Image")

    uploaded_file = st.file_uploader(
        "Upload a cell, X-ray, or pathology image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read and convert to grayscale; large photos are downscaled so six
        # full-resolution result figures fit in Streamlit Cloud's memory.
        pil_img = Image.open(uploaded_file).convert("L")
        if max(pil_img.size) > MAX_UPLOAD_SIDE:
            original_size = pil_img.size
            pil_img.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.Resampling.LANCZOS)
            st.caption(f"Resized from {original_size[0]}×{original_size[1]} to "
                       f"{pil_img.size[0]}×{pil_img.size[1]} for processing.")
        img = np.array(pil_img)

        st.image(
            pil_img,
            caption="Uploaded Medical Image",
            width=350
        )

        if st.button("Run Multi-Algorithm Segmentation"):
            with st.spinner("Running classical algorithms" + (" and Cellpose" if USE_CELLPOSE else "") + "..."):
                results = run_all(img, include_cellpose=USE_CELLPOSE)

            if has_bright_background(img):
                st.info(
                    "Detected a light background with dark cells (brightfield), so the image "
                    "was inverted before segmentation."
                )
            show_masks(results)
