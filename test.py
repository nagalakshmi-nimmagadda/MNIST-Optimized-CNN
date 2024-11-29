import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split
from model import Net
import sys

def test_model_accuracy():
    """Test model accuracy on the 10k test split"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Test transforms - only basic transforms, no augmentation
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full training dataset
    full_train_data = datasets.MNIST('../data', train=True, download=True, 
                                   transform=test_transforms)
    
    # Create 50k/10k split
    train_size = 50000
    test_size = len(full_train_data) - train_size
    _, test_data = random_split(full_train_data, 
                              [train_size, test_size],
                              generator=torch.Generator().manual_seed(42))
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    
    # Load model
    model = Net().to(device)
    try:
        model.load_state_dict(torch.load('mnist_model.pth'))
        print("Model loaded successfully!")
    except:
        print("Error: Could not load model weights. Make sure 'mnist_model.pth' exists.")
        return
    
    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f'\nTest Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

def check_model_requirements():
    """Verify that the model meets all requirements"""
    model = Net()
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    if total_params >= 20000:
        print("❌ Model has too many parameters (should be < 20,000)")
        return False
    else:
        print("✅ Parameter count check passed")
    
    # Check for BatchNorm
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    print("\nBatch Normalization layers:", 
          [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)])
    if not has_bn:
        print("❌ Model should use BatchNormalization")
        return False
    else:
        print("✅ BatchNorm check passed")
    
    # Check for Dropout
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    print("\nDropout layers:", 
          [m for m in model.modules() if isinstance(m, nn.Dropout)])
    if not has_dropout:
        print("❌ Model should use Dropout")
        return False
    else:
        print("✅ Dropout check passed")
    
    # Check for GAP or FC
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    print("\nGAP layers:", 
          [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)])
    print("FC layers:", 
          [m for m in model.modules() if isinstance(m, nn.Linear)])
    if not (has_gap or has_fc):
        print("❌ Model should use either Global Average Pooling or Fully Connected layer")
        return False
    else:
        print("✅ GAP/FC check passed")
    
    return True

def main():
    print("Checking model requirements...")
    if not check_model_requirements():
        print("\n❌ Model does not meet all requirements")
        sys.exit(1)
    
    print("\nTesting model accuracy...")
    accuracy = test_model_accuracy()
    
    if accuracy >= 99.4:
        print(f"✅ Model achieves required accuracy: {accuracy:.2f}%")
    else:
        print(f"❌ Model accuracy {accuracy:.2f}% is below required 99.4%")
        sys.exit(1)
    
    print("\n🎉 All checks passed! Model meets all requirements.")

if __name__ == "__main__":
    main() 