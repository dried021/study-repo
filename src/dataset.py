import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class MNISTCustomDataset(Dataset):
    """MNIST Custom Dataset"""
    
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        
        # 데이터 로드
        if train:
            data_file = os.path.join(data_dir, 'mnist_train.npz')
        else:
            data_file = os.path.join(data_dir, 'mnist_test.npz')
        
        data = np.load(data_file)
        self.images = data['images']
        self.labels = data['labels']
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_item_for_viz(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')
        
        viz_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
        image = viz_transform(image)
        
        return image, label


class CIFAR10CustomDataset(Dataset):
    """CIFAR-10 Custom Dataset"""
    
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        
        if train:
            data_file = os.path.join(data_dir, 'cifar10_train.npz')
        else:
            data_file = os.path.join(data_dir, 'cifar10_test.npz')
        
        data = np.load(data_file)
        self.images = data['images']
        self.labels = data['labels']
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_item_for_viz(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)
        
        viz_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        image = viz_transform(image)
        
        return image, label


def get_mnist_dataloaders(data_dir, batch_size=32, val_split=0.2, device='cpu'):
    """MNIST 데이터로더 생성"""
    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())
    
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_train_dataset = MNISTCustomDataset(
        data_dir=data_dir, 
        train=True, 
        transform=train_transform
    )
    
    test_dataset = MNISTCustomDataset(
        data_dir=data_dir, 
        train=False, 
        transform=test_transform
    )
    
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Windows에서는 0 권장
            pin_memory=use_pin_memory
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )
    }
    
    return dataloaders, test_dataset


def get_cifar_dataloaders(data_dir, batch_size=32, val_split=0.2, device='cpu'):
    """CIFAR-10 데이터로더 생성"""
    
    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())
    
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_train_dataset = CIFAR10CustomDataset(
        data_dir=data_dir, 
        train=True, 
        transform=train_transform
    )
    
    test_dataset = CIFAR10CustomDataset(
        data_dir=data_dir, 
        train=False, 
        transform=test_transform
    )
    
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # ✅ pin_memory 자동 설정
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Windows에서는 0 권장
            pin_memory=use_pin_memory
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )
    }
    
    return dataloaders, test_dataset