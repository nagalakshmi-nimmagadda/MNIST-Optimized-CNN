import torch
from model import Net
import sys

def check_model_requirements():
    try:
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
        has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
        print("\nBatch Normalization layers:", 
              [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
        if not has_bn:
            print("❌ Model should use BatchNormalization")
            return False
        else:
            print("✅ BatchNorm check passed")

        # Check for Dropout
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        print("\nDropout layers:", 
              [m for m in model.modules() if isinstance(m, torch.nn.Dropout)])
        if not has_dropout:
            print("❌ Model should use Dropout")
            return False
        else:
            print("✅ Dropout check passed")

        # Check for GAP or FC
        has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
        has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
        print("\nGAP layers:", 
              [m for m in model.modules() if isinstance(m, torch.nn.AvgPool2d)])
        print("FC layers:", 
              [m for m in model.modules() if isinstance(m, torch.nn.Linear)])
        if not (has_gap or has_fc):
            print("❌ Model should use either Global Average Pooling or Fully Connected layer")
            return False
        else:
            print("✅ GAP/FC check passed")

        print("\n🎉 All checks passed! Model meets all requirements.")
        return True

    except Exception as e:
        print(f"\n❌ Error during check: {str(e)}")
        return False

if __name__ == "__main__":
    success = check_model_requirements()
    sys.exit(0 if success else 1) 