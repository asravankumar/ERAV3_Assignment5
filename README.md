# MNIST Classification with CI/CD Pipeline


![Build Status](https://github.com/asravankumar/ERAV3_Assignment5/actions/workflows/ml-pipeline.yml/badge.svg?branch=main)

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
  - FC1: 16 * 7 * 7 → 25 neurons
  - FC2: 25 → 10 neurons (output layer)
- **Activation**: ReLU after each layer except the final output
- **Total Parameters**: < 25,000

## Training Details

- **Dataset**: MNIST
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Training Duration**: 1 epoch
- **Input Normalization**: Mean=0.1307, Std=0.3081

### Data Augmentation
The training pipeline includes random augmentations with 50% probability per image:
1. **Affine Transformations**:
   - Rotation: ±15 degrees
   - Translation: ±10% in both directions
   - Scaling: 90-110% of original size
2. **Pure Rotation**:
   - Random rotation up to ±20 degrees
3. **Gaussian Noise**:
   - Random noise with 0.1 standard deviation
   - Values clamped to [0,1] range

For each image that gets augmented (50% chance), one of these three techniques is randomly selected. Test/validation data remains unaugmented for consistent evaluation.

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs validation tests
5. Archives the trained model

### Validation Tests
1. **Model Architecture Tests**
   - Model parameter count (< 25,000)
   - Input/output dimensions (28x28 → 10 classes)

2. **Performance Tests**
   - Model accuracy (> 95% on test set)
   - Output probability distribution validation
   - Gradient flow verification

3. **Robustness Tests**
   - Zero input handling
   - All-ones input handling
   - Multiple batch size processing (1, 4, 16)

Each test ensures specific aspects of the model:
- `test_model_parameters`: Verifies model is lightweight
- `test_model_accuracy`: Checks performance on MNIST test set
- `test_input_output_dimensions`: Validates correct tensor shapes
- `test_output_probability_distribution`: Ensures valid probability outputs
- `test_model_gradient_flow`: Confirms proper backpropagation
- `test_model_on_zeros`: Checks model stability with zero inputs
- `test_model_on_ones`: Verifies handling of maximum value inputs
- `test_batch_processing`: Tests flexibility with different batch sizes

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