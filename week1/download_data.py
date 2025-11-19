import numpy as np
import os
from torchvision import datasets
from tqdm import tqdm

# CustomDataset을 정의하기 위해 데이터 다운로드 
def prepare_mnist_data(save_dir='./data'):
    # 데이터 저장 경로 지정

    print("Downloading MNIST dataset...")
    
    # 원본 데이터 다운로드
    train_dataset = datasets.MNIST(root='./temp', train=True, download=True)
    test_dataset = datasets.MNIST(root='./temp', train=False, download=True)
    
    # Train 데이터 변환
    print("Converting MNIST train data...")
    train_images = []
    train_labels = []
    
    for img, label in tqdm(train_dataset):
        # tqdm: python 진행률 프로세스바
        train_images.append(np.array(img)) # PIL Image 형식의 img를 numpy 배열로 변환
        train_labels.append(label)
    
    train_images = np.array(train_images) # list -> numpy 배열 (60000, 28, 28)
    train_labels = np.array(train_labels) # list -> (60000, )
    
    # Test 데이터 변환
    print("Converting MNIST test data...")
    test_images = []
    test_labels = []
    
    for img, label in tqdm(test_dataset):
        test_images.append(np.array(img))
        test_labels.append(label)
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # 저장
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving MNIST train data...")
    # npz로 압축해서 저장
    # 변수 train_images, train_labels 저장
    np.savez_compressed(
        os.path.join(save_dir, 'mnist_train.npz'),
        images=train_images,
        labels=train_labels
    )
    
    print("Saving MNIST test data...")
    np.savez_compressed(
        os.path.join(save_dir, 'mnist_test.npz'),
        images=test_images,
        labels=test_labels
    )
    
    print(f"MNIST data saved to {save_dir}")
    print(f"  - Train: {train_images.shape}, {train_labels.shape}")
    print(f"  - Test: {test_images.shape}, {test_labels.shape}")


def prepare_cifar10_data(save_dir='./data'):
    print("Downloading CIFAR-10 dataset...")
    
    # 원본 데이터 다운로드
    train_dataset = datasets.CIFAR10(root='./temp', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./temp', train=False, download=True)
    
    # Train 데이터 변환
    print("Converting CIFAR-10 train data...")
    train_images = []
    train_labels = []
    
    for img, label in tqdm(train_dataset):
        train_images.append(np.array(img))
        train_labels.append(label)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    # Test 데이터 변환
    print("Converting CIFAR-10 test data...")
    test_images = []
    test_labels = []
    
    for img, label in tqdm(test_dataset):
        test_images.append(np.array(img))
        test_labels.append(label)
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # 저장
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving CIFAR-10 train data...")
    np.savez_compressed(
        os.path.join(save_dir, 'cifar10_train.npz'),
        images=train_images,
        labels=train_labels
    )
    
    print("Saving CIFAR-10 test data...")
    np.savez_compressed(
        os.path.join(save_dir, 'cifar10_test.npz'),
        images=test_images,
        labels=test_labels
    )
    
    print(f"CIFAR-10 data saved to {save_dir}")
    print(f"  - Train: {train_images.shape}, {train_labels.shape}")
    print(f"  - Test: {test_images.shape}, {test_labels.shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare MNIST and CIFAR-10 datasets')
    parser.add_argument('--save_dir', type=str, default='./data',
                        help='Directory to save the processed data')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'both'],
                        default='both', help='Which dataset to prepare')
    
    args = parser.parse_args()
    
    if args.dataset in ['mnist', 'both']:
        prepare_mnist_data(args.save_dir)
        print("\n")
    
    if args.dataset in ['cifar10', 'both']:
        prepare_cifar10_data(args.save_dir)
    
    print("\nData preparation completed!")