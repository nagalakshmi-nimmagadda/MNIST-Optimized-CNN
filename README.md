# MNIST Classification with PyTorch

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification that achieves >99.4% test accuracy while meeting specific architectural constraints.

## Model Architecture Requirements

- Less than 20k Parameters
- Achieves 99.4% validation/test accuracy
- Trains in less than 20 Epochs
- Uses Batch Normalization
- Implements Dropout
- Uses either Global Average Pooling or Fully Connected layer (current implementation uses both)

## Model Architecture Details

The model follows a structured architecture with:
1. Input Block (1→8 channels)
2. First Convolution Block (8→16 channels)
3. Transition Block with MaxPooling
4. Second and Third Convolution Blocks (maintaining 16 channels)
5. Another Transition Block
6. Fourth Convolution Block (16→32 channels)
7. Global Average Pooling
8. Final Fully Connected Layer (32→10)

Key features:
- BatchNorm after each convolution
- Dropout (0.1) for regularization
- 3x3 convolutions throughout
- Two MaxPooling layers for spatial reduction
- Combination of GAP and FC layer at the end

## Project Structure

```
├── model.py # Model architecture definition
├── train.py # Training and evaluation code
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
- Training Time: 15-18 epochs

## Model Training Logs

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

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

