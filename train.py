import torch
from torch import nn
from src.vision_cnn import (
    get_device, 
    build_model, 
    get_fashion_mnist_loaders, 
    train_one_epoch, 
    evaluate
)

def main():
    print("1. Detecting hardware and downloading the FashionMNIST dataset...")
    device = get_device()

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=64)

    print(f"2. Building the neural network on {device}...")
    model, _ = build_model()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5 
    print(f"3. Starting training. Total epochs: {epochs}...")
    
    for t in range(epochs):
        print(f"Epoch {t+1} / {epochs}")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"   Train Loss: {train_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")


    save_path = "fashion_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"4. Done! Model weights have been saved to: {save_path}")

if __name__ == "__main__":
    main()