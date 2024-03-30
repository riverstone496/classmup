import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # 画像データ
        label = self.targets[idx]  # 対応するラベル
        return image, label  # 画像とラベルのタプルを返す

class FilteredCIFAR10(Dataset):
    def __init__(self, cifar10_dataset, classes_to_keep):
        self.data = []
        self.targets = []
        
        for img, label in cifar10_dataset:
            if label in classes_to_keep:
                self.data.append(img)
                self.targets.append(label)
        
        self.data = torch.stack(self.data)  # Convert list of PIL images to tensor
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

def save_filtered_cifar10(classes_to_keep, train, save_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    cifar10_dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    filtered_dataset = FilteredCIFAR10(cifar10_dataset, classes_to_keep)
    
    torch.save((filtered_dataset.data, filtered_dataset.targets), save_path)

# 最初の8クラスと最後の2クラスのデータセットをフィルタリングして保存
save_filtered_cifar10(list(range(8)), train=True, save_path='./data/splitCIFAR10/CIFAR10_8_train.pt')
save_filtered_cifar10(list(range(8)), train=False, save_path='./data/splitCIFAR10/CIFAR10_8_test.pt')
save_filtered_cifar10(list(range(8, 10)), train=True, save_path='./data/splitCIFAR10/CIFAR10_2_train.pt')
save_filtered_cifar10(list(range(8, 10)), train=False, save_path='./data/splitCIFAR10/CIFAR10_2_test.pt')

def load_dataset(file_path):
    data, targets = torch.load(file_path)
    return CustomDataset(data, targets)

# 使用例
train_dataset = load_dataset('./data/splitCIFAR10/CIFAR10_8_train.pt')

# DataLoaderの定義
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for (images, labels) in train_loader:
    # ここでトレーニングのロジックを実装
    print(labels)
    pass
