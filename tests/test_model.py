import torch
import pytest
from model.network import SimpleCNN
from torchvision import datasets, transforms

def test_model_parameters():
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    try:
        # Load the latest model
        import glob
        import os
        model_files = glob.glob('models/*.pth')
        if not model_files:
            pytest.skip("No model file found to test accuracy")
            
        latest_model = max(model_files, key=os.path.getctime)
        model.load_state_dict(torch.load(latest_model, weights_only=True))
        
        # Load test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
        
        # Evaluate accuracy
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Model accuracy is {accuracy:.2f}%")
        assert accuracy > 95, f"Model accuracy is {accuracy:.2f}%, should be > 95%"
        
    except Exception as e:
        pytest.fail(f"Error during accuracy testing: {str(e)}")

def test_input_output_dimensions():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_output_probability_distribution():
    """Test if model outputs valid probability distributions"""
    model = SimpleCNN()
    model.eval()
    test_input = torch.randn(5, 1, 28, 28)
    with torch.no_grad():
        output = torch.softmax(model(test_input), dim=1)
        # Check if probabilities sum to 1
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5), "Output probabilities don't sum to 1"
        # Check if all probabilities are between 0 and 1
        assert (output >= 0).all() and (output <= 1).all(), "Output contains invalid probabilities"

def test_model_gradient_flow():
    """Test if gradients are flowing through the model properly"""
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    test_target = torch.tensor([5])  # Random target class
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    output = model(test_input)
    loss = criterion(output, test_target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not zero for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Zero gradient for {name}"


def test_model_on_zeros():
    """Test if model can handle an input of all zeros"""
    model = SimpleCNN()
    model.eval()
    
    # Create an input tensor of zeros
    zero_input = torch.zeros(1, 1, 28, 28)
    
    try:
        output = model(zero_input)
        assert output.shape == (1, 10), "Model failed to process zero input"
        assert not torch.isnan(output).any(), "Model produced NaN values for zero input"
    except Exception as e:
        pytest.fail(f"Model failed on zero input: {str(e)}")

def test_model_on_ones():
    """Test if model can handle an input of all ones"""
    model = SimpleCNN()
    model.eval()
    
    # Create an input tensor of ones
    ones_input = torch.ones(1, 1, 28, 28)
    
    try:
        output = model(ones_input)
        assert output.shape == (1, 10), "Model failed to process ones input"
        assert not torch.isnan(output).any(), "Model produced NaN values for ones input"
    except Exception as e:
        pytest.fail(f"Model failed on ones input: {str(e)}")

def test_batch_processing():
    """Test if model can handle different batch sizes"""
    model = SimpleCNN()
    model.eval()
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 16]
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (batch_size, 10), f"Failed to process batch size {batch_size}"
        print(f"Successfully processed batch size: {batch_size}")