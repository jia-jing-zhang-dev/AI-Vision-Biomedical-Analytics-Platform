# AI Vision & Biomedical Analytics Platform

An educational machine-learning, computer-vision, and biomedical-image-analysis project. It includes a Streamlit interface and reusable modules for classification, clustering, image segmentation, convolution, visualization, and deep learning.

## Features

- **Blood-cell classification** — a PyTorch CNN that classifies 8 blood-cell types from BloodMNIST at **96.6% test accuracy** (vs. 72.9% for a fully connected baseline), with hyperparameters tuned by Optuna on the validation split.
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
│   ├── 2_Biomedical_Segmentation.py # Biomedical segmentation page
│   └── 3_Blood_Cell_Classifier.py   # BloodMNIST classification page
├── src/
│   ├── vision_cnn.py                # PyTorch model and training helpers
│   ├── medical_cnn.py               # BloodMNIST CNN, training and inference helpers
│   ├── segmentation.py              # Classical segmentation and Dice / IoU evaluation
│   ├── classification.py            # Supervised learning and Optuna tuning
│   ├── clustering.py                # K-Means and silhouette-based model selection
│   ├── convolution.py               # NumPy 2D convolution
│   └── visualization.py             # EDA visualization helpers
├── train.py                         # FashionMNIST training script
├── train_medical.py                 # BloodMNIST training / tuning script
├── fashion_model.pth                # Trained FashionMNIST weights
├── biomedical_cnn.pth               # Trained BloodMNIST CNN weights
├── biomedical_cnn_results.json      # Hyperparameters and val/test accuracy
├── requirements.txt                 # Base dependencies
└── tests/                          # pytest suite
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
3. **Blood Cell Classifier** — pick a bundled BloodMNIST test image (one per class) or upload your own, and see the CNN's prediction and class probabilities.

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

The script downloads BloodMNIST from MedMNIST, trains the CNN for 15 epochs, keeps the epoch with the best **validation** accuracy, evaluates the held-out **test** split once, and writes `biomedical_cnn.pth`, `biomedical_cnn_results.json` and `public/bloodmnist-confusion.png`. The Blood Cell Classifier page loads these files.

```bash
python train_medical.py --optuna 20   # tune lr, weight decay, dropout and width on the validation split first
python train_medical.py --model mlp   # reproduce the original fully connected baseline
```

#### Results (BloodMNIST test split, 3,421 images)

| Model | Test accuracy |
|---|---|
| Fully connected baseline (flattened pixels → 128 → 8, 5 epochs) | 72.9% |
| CNN, default hyperparameters (3 seeds) | 96.1–96.6% |
| **CNN, Optuna-tuned (20 trials) — shipped model** | **96.6%** |

Most of the gain comes from switching to a convolutional architecture with augmentation (random flips and 90° rotations); Optuna tuning adds little on top, within seed-to-seed variation.

![BloodMNIST confusion matrix](public/bloodmnist-confusion.png)

The hardest classes are immature granulocytes and monocytes (92% recall each), which are mostly confused with each other.

*Data: [MedMNIST v2](https://medmnist.com/) BloodMNIST, CC BY 4.0. Educational project — not for clinical use.*

## Run Tests

```bash
pytest tests/
```

The tests cover classification, clustering, segmentation metrics, 2D convolution, and the BloodMNIST CNN (output shapes, inference, and a check that the shipped model correctly classifies the bundled sample images).

## Troubleshooting

### `fashion_model.pth` cannot be found

Run `python train.py` from the project root, or confirm that the trained weight file is still located there.

### Dataset download fails during training

Both `train.py` and `train_medical.py` download their datasets on the first run. Check your internet connection and try again.

### Uploaded clothing images produce weak predictions

The model is trained on 28×28 grayscale FashionMNIST images. For better results, use a clear, centered image of a single clothing item. The application converts images to grayscale, resizes them, and inverts their colors before inference.

