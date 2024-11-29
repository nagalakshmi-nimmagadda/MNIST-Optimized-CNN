import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Initial Feature Extraction
        self.conv1 = nn.Conv2d(1, 10, 3)  # 26x26x10
        self.bn1 = nn.BatchNorm2d(10)
        
        # Feature Processing Block 1
        self.conv2 = nn.Conv2d(10, 16, 3)  # 24x24x16
        self.bn2 = nn.BatchNorm2d(16)
        
        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)  # 12x12x16
        self.dropout1 = nn.Dropout(0.008)
        
        # Feature Processing Block 2
        self.conv3 = nn.Conv2d(16, 16, 3)  # 10x10x16
        self.bn3 = nn.BatchNorm2d(16)
        
        # Feature Processing Block 3
        self.conv4 = nn.Conv2d(16, 20, 3)  # 8x8x20
        self.bn4 = nn.BatchNorm2d(20)
        
        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2)  # 4x4x20
        self.dropout2 = nn.Dropout(0.008)
        
        # Final Feature Processing
        self.conv5 = nn.Conv2d(20, 32, 3)  # 2x2x32
        self.bn5 = nn.BatchNorm2d(32)
        
        # Global Average Pooling and Classification
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x32
        self.dropout3 = nn.Dropout(0.008)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # Initial features with residual
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First block
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Second and third blocks
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))
        
        # Final processing
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = self.dropout3(x.view(-1, 32))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
