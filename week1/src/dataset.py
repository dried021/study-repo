import torch
from torch.utils.data import Dataset, DataLoader
# Dataset: 샘플과 해당 레이블을 저장
# DataLoader: 반복 가능한 객체로 래핑
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
class MNISTCustomDataset(Dataset):
    """MNIST Custom Dataset"""
    # Dataset을 상속 받아 dataset을 구성
    # __init__, __len__, __getitem__ 함수 구현 필요
    
    def __init__(self, data_dir, train=True, transform=None):
        # 이 class의 속성 및 메소드 정의
        # Dataset 객체를 인스턴스화할 때 한 번 실행
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        
        # 데이터 로드
        if train:
            data_file = os.path.join(data_dir, 'mnist_train.npz')
        else:
            data_file = os.path.join(data_dir, 'mnist_test.npz')
        
        data = np.load(data_file)
        # np.savez_compressed를 통해 numpy.ndarray 저장한 것을 로드

        self.images = data['images']
        # colab에서 실행할 때는 괜찮은데 로컬에서 실행할 때 int 32로 나와서 int 64로 변환..
        self.labels = data['labels'].astype(np.int64)
        
    def __len__(self):
        # 데이터 세트의 크기 반환
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 모든 이미지가 한 번에 메모리에 저장되지 않고 필요에 따라 읽힘 -> 메모리 효율성 높음
        # 주어진 인덱스의 데이터 세트의 샘플 로드, 반환
        # 텐서 이미지와 해당 레이블을 튜플로 반환
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            # transform이 정의되었다면 transform 실행
            image = self.transform(image)
        
        return image, label
    
    def get_item_for_viz(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')
        
        # 보기 좋은 시각화를 위한 transform
        # 정규화를 실행하지 않음
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
        self.labels = data['labels'].astype(np.int64)
        
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


# Dataloader
# 배치 크기의 feature와 label을 구성해서 반환. 반복 가능한 객체
def get_mnist_dataloaders(data_dir, batch_size=32, val_split=0.2, device='cpu'):
    """MNIST 데이터로더 생성"""
    # pin memory 설정: gpu에서 true로 설정할 경우 cpu와 gpu 간에 더 빠르게 복사되어 학습 속도를 향상시킬 수 있음
    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())
    
    # 변환 정의
    # Resize
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
    
    # len(dataset) 기반으로 train/val 크기를 조정
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # 데이터를 train과 validation으로 분리
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # pin_memory: GPU 메모리에 데이터를 고정할지 여부
    # GPU를 사용할 경우 TRUE
    # CPU 환경에서 실행했을 때 pin_memory=True로 설정되어 오류가 나서 추가
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
    '''
    dataset: 데이터를 로드할 데이터셋 객체 지정. 
           torch.utils.data.Dataset 클래스를 상속받은 사용자 정의 데이터셋 클래스 or torchvision의 내장 데이터셋 클래스 사용
    batch_size: 한 번에 로드할 batch 크기 지정
    shuffle: 데이터를 섞을지 여부. True일 경우 데이터가 매 epoch마다 섞임
    num_workers: 데이터 로딩에 사용할 worker수 지정
    pin_momory: GPU 메모리에 데이터를 고정할지 여부
    collate_fn: 배치를 생성하기 전 데이터를 결합하는 함수. 데이터셋이 출력하는 원시 데이터의 리스트를 배치로 결합할 수 있음
        예를 들어 같은 배치에서 input의 크기가 다를 수 있을 경우 zero padding을 해주는 함수를 지정할 수 있음
    drop_out: 마지막 배치의 크기가 batch_size보다 작을 경우 해당 배치를 무시할지 여부
    '''
    
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