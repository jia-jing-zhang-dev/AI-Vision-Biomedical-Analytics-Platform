# AI Vision & Biomedical Analytics Platform

An educational machine-learning, computer-vision, and biomedical-image-analysis project. It includes a Streamlit interface and reusable modules for classification, clustering, image segmentation, convolution, visualization, and deep learning.

## Features

- **FashionMNIST clothing classification** — a PyTorch three-layer fully connected network that predicts one of ten clothing classes from an uploaded image.
- **Biomedical image segmentation** — five classical methods: Otsu thresholding, adaptive thresholding, K-Means, Canny edge filling, and Watershed.
- **Segmentation evaluation** — generates annotated synthetic biomedical images and reports Dice and IoU metrics.
- **Machine-learning utilities** — decision trees, random forests, ANOVA feature selection, Optuna tuning, and K-Means clustering analysis.
- **Image-processing fundamentals** — a NumPy implementation of 2D convolution with edge-detection, sharpening, Sobel, and box-blur kernels.
- **Visualization utilities** — EDA plotting helpers built with Matplotlib and Seaborn.

## Project Structure

```text
.
├── app.py                          # Streamlit home page
├── pages/
│   ├── 1_Deep_Learning.py           # FashionMNIST classification page
│   └── 2_Biomedical_Segmentation.py # Biomedical segmentation page
├── src/
│   ├── vision_cnn.py                # PyTorch model and training helpers
│   ├── segmentation.py              # Classical segmentation and Dice / IoU evaluation
│   ├── classification.py            # Supervised learning and Optuna tuning
│   ├── clustering.py                # K-Means and silhouette-based model selection
│   ├── convolution.py               # NumPy 2D convolution
│   └── visualization.py             # EDA visualization helpers
├── train.py                         # FashionMNIST training script
├── train_medical.py                 # BloodMNIST training script
├── fashion_model.pth                # Trained FashionMNIST weights
├── biomedical_model.pth             # Trained BloodMNIST weights
├── requirements.txt                 # Base dependencies
└── tests/test_modules.py            # Module smoke tests
```

## Requirements

- Python 3.10 or later
- A virtual environment is recommended

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the base dependencies:

```bash
pip install -r requirements.txt
```

Install the following additional packages to run biomedical classification training or the test suite:

```bash
pip install medmnist pytest
```

> PyTorch installation differs by CPU, CUDA version, and operating system. If the default installation does not work, use the command recommended on the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

## Run the Web Application

From the project root, run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, then use the sidebar to switch between modules:

1. **Deep Learning** — upload a JPG, JPEG, or PNG clothing image and run FashionMNIST classification.
2. **Biomedical Segmentation** — either generate a synthetic benchmark image to compare masks and Dice / IoU scores, or upload a grayscale or color medical image to view segmentation results from every algorithm.

## Train Models

### FashionMNIST Clothing Classifier

```bash
python train.py
```

The script downloads FashionMNIST to `data/`, trains for five epochs, and saves the weights as `fashion_model.pth`. The classification page loads this file; if it is removed or replaced, train the model again before starting the page.

### BloodMNIST Biomedical Classifier

```bash
python train_medical.py
```

The script downloads the BloodMNIST dataset from MedMNIST, trains an eight-class fully connected classifier for five epochs, and writes `biomedical_model.pth`.

> The current Streamlit biomedical module demonstrates **classical image segmentation** and does not load `biomedical_model.pth`. The weight file is available for future biomedical-classification pages or offline inference.

## Run Tests

```bash
pytest tests/
```

The smoke tests cover classification, clustering, segmentation metrics, and 2D convolution.

## Troubleshooting

### `fashion_model.pth` cannot be found

Run `python train.py` from the project root, or confirm that the trained weight file is still located there.

### Dataset download fails during training

Both `train.py` and `train_medical.py` download their datasets on the first run. Check your internet connection and try again.

### Uploaded clothing images produce weak predictions

The model is trained on 28×28 grayscale FashionMNIST images. For better results, use a clear, centered image of a single clothing item. The application converts images to grayscale, resizes them, and inverts their colors before inference.

## License

No license file is currently included. Add an appropriate open-source license before publishing or redistributing the project.
