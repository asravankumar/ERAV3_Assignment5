# MNIST Classification with CI/CD Pipeline

This project implements a simple Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The pipeline includes automated training, testing, and model validation.

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── model/
│   ├── __init__.py
│   └── network.py
├── tests/
│   └── test_model.py
├── train.py
├── requirements.txt
└── .gitignore
```

## Model Architecture

The model (`SimpleCNN`) is a lightweight convolutional neural network designed for MNIST digit classification:

- **Input Layer**: Accepts 28x28 grayscale images
- **Convolutional Layers**:
  - Conv1: 1 → 10 channels (3x3 kernel, padding=1)
  - Conv2: 10 → 16 channels (3x3 kernel, padding=1)
- **Pooling**: MaxPool2d (2x2) after each conv layer
- **Fully Connected Layers**:
  - FC1: 16 * 7 * 7 → 32 neurons
  - FC2: 32 → 10 neurons (output layer)
- **Activation**: ReLU after each layer except the final output
- **Total Parameters**: < 100,000

## Training Details

- **Dataset**: MNIST
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Training Duration**: 1 epoch
- **Input Normalization**: Mean=0.1307, Std=0.3081

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs validation tests
5. Archives the trained model

### Validation Tests
- Model parameter count (< 100,000)
- Input/output dimensions (28x28 → 10 classes)
- Model accuracy (> 80% on test set)

## Local Development

### Setup
1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally
1. Train the model:
```bash
python train.py
```

2. Run tests:
```bash
pytest tests/test_model.py -v
```

## Model Artifacts

Trained models are saved in the `models/` directory with timestamps in the format:
```
model_YYYYMMDD_HHMMSS.pth
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest

## Notes

- The model is designed to be lightweight while maintaining >80% accuracy on MNIST
- Training progress shows batch-wise loss and accuracy
- Model checkpoints include timestamps for version tracking
- All tests must pass for successful CI/CD pipeline completion

## Future Improvements

- Add model versioning
- Implement early stopping
- Add data augmentation
- Extend to other datasets
- Add model performance visualization

This README provides:
1. Clear project overview
2. Detailed model architecture
3. Training specifications
4. Setup and usage instructions
5. CI/CD pipeline details
6. Future improvement suggestions

It should help new users understand and use the project effectively.