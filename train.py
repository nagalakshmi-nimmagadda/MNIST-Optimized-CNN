import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
from model import Net
from tqdm import tqdm

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(
            desc=f'Epoch {epoch} Loss={loss.item():.4f} Batch_Acc={100*correct/processed:.2f}%'
        )

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Essential augmentation only
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-5.0, 5.0), fill=(0,)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.98, 1.02)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load and split data
    full_train_data = datasets.MNIST('../data', train=True, download=True, 
                                   transform=train_transforms)
    
    # Create 50k/10k split
    train_size = 50000
    test_size = len(full_train_data) - train_size
    train_data, test_data = random_split(full_train_data, 
                                       [train_size, test_size],
                                       generator=torch.Generator().manual_seed(42))
    
    test_data.dataset.transform = test_transforms

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, 
                                             shuffle=True, num_workers=2,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

    model = Net().to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=15,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    best_acc = 0
    for epoch in range(1, 16):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        test_acc = test(model, device, test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'mnist_model.pth')
            print(f'New best test accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
