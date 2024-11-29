# MNIST Classification with PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)](../../actions)


[![Accuracy](https://img.shields.io/badge/Accuracy-99.43%25-brightgreen.svg?style=flat-square)](../../)
[![Parameters](https://img.shields.io/badge/Parameters-13.1K-brightgreen.svg?style=flat-square)](../../)
[![Training Time](https://img.shields.io/badge/Epochs-15-brightgreen.svg?style=flat-square)](../../)

A highly optimized PyTorch implementation of MNIST digit classification achieving 99.42% accuracy with less than 20k parameters.

## 🎯 Performance Metrics

| Metric | Target | Achieved |
|:-------|:------:|:--------:|
| Test Accuracy | ≥99.4% | **99.43%** |
| Parameter Count | <20k | **13,086** |
| Training Epochs | <20 | **15** |

## 🏗️ Architecture Overview

### Core Components
- BatchNorm after each convolution ✓
- Lightweight Dropout (0.008) ✓
- Global Average Pooling + FC layer ✓

### Network Structure
<details>
<summary>Click to expand detailed architecture</summary>

1. **Initial Feature Extraction**
   - Conv2d(1, 10, 3) → BatchNorm2d → ReLU
   - Output: 26x26x10

2. **Feature Processing Block 1**
   - Conv2d(10, 16, 3) → BatchNorm2d → ReLU
   - Output: 24x24x16

3. **Transition Block 1**
   - MaxPool2d(2, 2)
   - Dropout(0.008)
   - Output: 12x12x16

4. **Feature Processing Block 2**
   - Conv2d(16, 16, 3) → BatchNorm2d → ReLU
   - Output: 10x10x16

5. **Feature Processing Block 3**
   - Conv2d(16, 20, 3) → BatchNorm2d → ReLU
   - Output: 8x8x20

6. **Transition Block 2**
   - MaxPool2d(2, 2)
   - Dropout(0.008)
   - Output: 4x4x20

7. **Final Feature Processing**
   - Conv2d(20, 32, 3) → BatchNorm2d → ReLU
   - Output: 2x2x32

8. **Classification**
   - Global Average Pooling
   - Dropout(0.008)
   - Linear(32, 10)
   - LogSoftmax
</details>

### Channel Progression
1 → 10 → 16 → 16 → 20 → 32

## Training Configuration

### Data Augmentation

## Project Structure

```
├── model.py # Model architecture definition
├── train.py # Training and evaluation code
├── test.py # test file
├── check_model_local.py # Local testing script
└── .github
└── workflows
└── model_check.yml # GitHub Actions workflow
```


## Requirements

- Python 3.8+
- PyTorch
- torchvision
- tqdm

Install dependencies:
pip install torch torchvision tqdm


## Usage

### Local Testing
To check if the model meets all requirements locally:
python check_model_local.py


### Training
To train the model:

The training script includes:
- Data augmentation (random rotation)
- Learning rate scheduling
- Model checkpointing
- Progress bars with tqdm
- Validation after each epoch

### Training Parameters
- Optimizer: Adam
  - Learning Rate: 0.001
  - Weight Decay: 5e-5
- Scheduler: OneCycleLR
  - Max LR: 0.003
  - Epochs: 15
  - pct_start: 0.2
  - anneal_strategy: 'cos'
  - div_factor: 25.0
  - final_div_factor: 1000.0
- Batch Size: 128
- Data Split: 50,000 training / 10,000 testing


## Implementation Details

### Key Features
1. No padding in convolutions for better feature extraction
2. Very light dropout (0.008) to maintain feature quality
3. BatchNorm after every convolution
4. Progressive spatial reduction through convolution strides
5. Efficient channel width progression
6. Global Average Pooling followed by single FC layer


## 📈 Results and Test Logs

Training set size: 50000
Test set size: 10000
Total parameters: 13,086
Epoch 1 Loss=0.3595 Batch_Acc=73.92%: 100%|███████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.31it/s]

Test set: Average loss: 0.3374, Accuracy: 9641/10000 (96.41%)

New best test accuracy: 96.41%
Epoch 2 Loss=0.0432 Batch_Acc=97.92%: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.48it/s] 

Test set: Average loss: 0.0791, Accuracy: 9783/10000 (97.83%)

New best test accuracy: 97.83%
Epoch 3 Loss=0.0588 Batch_Acc=98.69%: 100%|███████████████████████████████████████████████████████| 391/391 [01:02<00:00,  6.24it/s] 

Test set: Average loss: 0.0508, Accuracy: 9837/10000 (98.37%)

New best test accuracy: 98.37%
Epoch 4 Loss=0.0339 Batch_Acc=98.88%: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.48it/s] 

Test set: Average loss: 0.0443, Accuracy: 9857/10000 (98.57%)

New best test accuracy: 98.57%
Epoch 5 Loss=0.0355 Batch_Acc=99.14%: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.49it/s] 

Test set: Average loss: 0.0337, Accuracy: 9890/10000 (98.90%)

New best test accuracy: 98.90%
Epoch 6 Loss=0.0447 Batch_Acc=99.27%: 100%|███████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.54it/s] 

Test set: Average loss: 0.0285, Accuracy: 9914/10000 (99.14%)

New best test accuracy: 99.14%
Epoch 7 Loss=0.0069 Batch_Acc=99.32%: 100%|███████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s] 

Test set: Average loss: 0.0271, Accuracy: 9914/10000 (99.14%)

Epoch 8 Loss=0.0061 Batch_Acc=99.53%: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.50it/s] 

Test set: Average loss: 0.0291, Accuracy: 9902/10000 (99.02%)

Epoch 9 Loss=0.0101 Batch_Acc=99.61%: 100%|███████████████████████████████████████████████████████| 391/391 [11:22<00:00,  1.74s/it] 

Test set: Average loss: 0.0294, Accuracy: 9907/10000 (99.07%)

Epoch 10 Loss=0.0035 Batch_Acc=99.71%: 100%|██████████████████████████████████████████████████████| 391/391 [01:22<00:00,  4.74it/s] 

Test set: Average loss: 0.0238, Accuracy: 9925/10000 (99.25%)

New best test accuracy: 99.25%
Epoch 11 Loss=0.0026 Batch_Acc=99.76%: 100%|██████████████████████████████████████████████████████| 391/391 [01:09<00:00,  5.64it/s] 

Test set: Average loss: 0.0218, Accuracy: 9934/10000 (99.34%)

New best test accuracy: 99.34%
Epoch 12 Loss=0.0097 Batch_Acc=99.87%: 100%|██████████████████████████████████████████████████████| 391/391 [01:11<00:00,  5.49it/s] 

Test set: Average loss: 0.0197, Accuracy: 9943/10000 (99.43%)

New best test accuracy: 99.43%
Epoch 13 Loss=0.0005 Batch_Acc=99.92%: 100%|██████████████████████████████████████████████████████| 391/391 [01:08<00:00,  5.68it/s] 

Test set: Average loss: 0.0200, Accuracy: 9947/10000 (99.47%)

New best test accuracy: 99.47%
Epoch 14 Loss=0.0027 Batch_Acc=99.93%: 100%|██████████████████████████████████████████████████████| 391/391 [01:04<00:00,  6.04it/s] 

Test set: Average loss: 0.0205, Accuracy: 9944/10000 (99.44%)

Epoch 15 Loss=0.0030 Batch_Acc=99.94%: 100%|██████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.37it/s] 

Test set: Average loss: 0.0197, Accuracy: 9943/10000 (99.43%)

## Model Performance

- Target Accuracy: >99.4%
- Parameters: ~15k
- Training Time: <20 epochs
- Dataset Split: 50k training, 10k testing

## Implementation Details

### Key Components:
1. **Batch Normalization**: Applied after each convolution layer to stabilize training
2. **Dropout**: Used throughout (0.1 rate) to prevent overfitting
3. **MaxPooling**: Strategic placement for spatial reduction
4. **Global Average Pooling**: Reduces parameters while maintaining performance
5. **Learning Rate Scheduler**: ReduceLROnPlateau for optimal convergence

### Data Augmentation:
- Random rotation (-7° to +7°)
- Normalization with MNIST mean (0.1307) and std (0.3081)

### Training Configuration:
- Optimizer: SGD with momentum (0.9)
- Initial Learning Rate: 0.01
- Batch Size: 128
- Loss Function: Negative Log Likelihood

## GitHub Actions

The repository includes automated checks through GitHub Actions that verify:
1. Parameter count is under 20k
2. Presence of BatchNormalization
3. Presence of Dropout
4. Presence of either GAP or FC layer

## Results

The model achieves:
- Test Accuracy: >99.4%
- Parameter Count: ~15k
- Training Time: 15

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

