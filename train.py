import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
import random

class RandomAugmentation:
    """Custom augmentation that randomly applies different transformations"""
    def __init__(self, p=0.7):
        self.p = p  # Probability of applying any augmentation
        
        # Define individual transforms
        self.affine = transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=0
        )
        self.rotate = transforms.RandomRotation(20, fill=0)
        
    def add_noise(self, img):
        """Add random Gaussian noise"""
        noise = torch.randn_like(img) * 0.1
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0., 1.)
    
    def __call__(self, img):
        # With probability 1-p, return original image
        if random.random() > self.p:
            return img
            
        # Randomly choose one of three augmentations
        aug_type = random.choice(['affine', 'rotate', 'noise'])
        
        if aug_type == 'affine':
            return self.affine(img)
        elif aug_type == 'rotate':
            return self.rotate(img)
        else:  # noise
            return self.add_noise(img)

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced transform pipeline with random augmentations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        RandomAugmentation(p=0.5),  # Apply random augmentation with 50% probability
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Simple transform for test/validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets with different transforms
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy for this batch
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        accuracy = 100 * correct / len(target)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/model_{timestamp}_acc{test_accuracy:.1f}.pth')
    
if __name__ == "__main__":
    train() 