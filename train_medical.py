import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

# 1. Image preprocessing: convert images to PyTorch tensors
# and normalize them
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 2. Load the biomedical dataset
# BloodMNIST: blood cell classification with 3-channel RGB images
data_flag = "bloodmnist"
info = INFO[data_flag]
DataClass = getattr(medmnist, info["python_class"])

n_channels = info["n_channels"]     # BloodMNIST has 3 channels
n_classes = len(info["label"])      # BloodMNIST contains 8 blood cell classes

print(
    f"Loading biomedical dataset: {data_flag} "
    f"(Channels: {n_channels}, Classes: {n_classes})..."
)

train_dataset = DataClass(
    split="train",
    transform=data_transform,
    download=True
)

test_dataset = DataClass(
    split="test",
    transform=data_transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

# 3. Define a fully connected classification network
# Input dimension: n_channels * 28 * 28 = 2352
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(n_channels * 28 * 28, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, n_classes)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting biomedical image classification model training...")

# 4. Train for 5 epochs
for epoch in range(5):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Convert labels to a 1D tensor
        labels = labels.squeeze().long()
        
        optimizer.zero_grad()
        outputs = model(images.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(
        f"Epoch [{epoch + 1}/5], "
        f"Loss: {running_loss / len(train_loader):.4f}"
    )

# 5. Save the trained biomedical model weights
torch.save(model.state_dict(), "biomedical_model.pth")

print(
    "🎉 Training completed successfully! "
    "The trained 'biomedical_model.pth' weight file "
    "has been generated in the project root directory."
)