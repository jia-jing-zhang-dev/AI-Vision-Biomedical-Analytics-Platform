"""End-to-End Machine Learning & Vision Analytics Framework.

Modules:
    visualization  - EDA plotting utilities
    classification - supervised learning (Decision Tree / Random Forest / Optuna tuning)
    clustering     - unsupervised learning (K-Means + silhouette model selection)
    segmentation   - classical CV image segmentation + Dice/IoU benchmarking
    convolution    - from-scratch 2D convolution & edge-detection kernels
    vision_cnn     - PyTorch feed-forward network for image classification
    medical_cnn    - PyTorch CNN for BloodMNIST blood-cell classification
    image_io       - upload loading/normalization (16-bit, transparency, EXIF, size limits)
    bbbc039        - BBBC039 real nuclei dataset loader (mask decoding, normalization)
    unet           - U-Net for nucleus instance segmentation (3-class formulation)
    explain        - Grad-CAM for the blood-cell classifiers
    calibration    - temperature scaling, ECE and reliability curves
    runtime        - app resource settings (heavy-work lock, torch thread limit)
    blood_cells    - finds and crops stained cells in blood-smear uploads
"""

__version__ = "1.2.0"
