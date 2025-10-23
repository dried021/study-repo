import torch
import torch.nn as nn
import torch.optim as optim
import os

from config import *
from src.models import initialize_model, get_params_to_update
from src.train import train
from src.visualization import plot_training_history
from src.dataset import get_mnist_dataloaders, get_cifar_dataloaders

# 데이터로더 생성

os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def main():
    print(f"Using device: {device}")
    print(f"Feature extraction mode: {feature_extract}")
    print("-" * 50)

    mnist_dataloaders, mnist_test = get_mnist_dataloaders(data_dir, batch_size, device=str(device))
    cifar_dataloaders, cifar_test = get_cifar_dataloaders(data_dir, batch_size, device=str(device))

    
    # ==================== MNIST 학습 ====================
    print("\n[1] Training MNIST AlexNet...")
    print("=" * 50)
    
    # 모델 초기화
    mnist_alexnet, input_size = initialize_model(
        "alexnet", num_classes, feature_extract, use_pretrained
    )
    mnist_alexnet = mnist_alexnet.to(device)
    
    # 학습 파라미터 설정
    params_to_update = get_params_to_update(mnist_alexnet, feature_extract)
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    mnist_alexnet, mnist_history = train(
        mnist_alexnet, mnist_dataloaders, criterion, 
        optimizer, device, num_epochs=num_epochs
    )
    
    # 모델 저장
    torch.save({
        'model_state_dict': mnist_alexnet.state_dict(),
        'history': mnist_history,
        'epoch': num_epochs,
        'best_acc': max(mnist_history['val_acc'])
    }, f'{save_dir}/mnist_alexnet_best.pth')
    print(f"MNIST model saved to {save_dir}/mnist_alexnet_best.pth")
    
    # 학습 결과 시각화
    plot_training_history(
        mnist_history, 
        'MNIST', 
        save_path=f'{results_dir}/mnist_training_history.png'
    )
    
    # ==================== CIFAR-10 학습 ====================
    print("\n[2] Training CIFAR-10 AlexNet...")
    print("=" * 50)
    
    # 모델 초기화
    cifar_alexnet, input_size = initialize_model(
        "alexnet", num_classes, feature_extract, use_pretrained
    )
    cifar_alexnet = cifar_alexnet.to(device)
    
    # 학습 파라미터 설정
    params_to_update = get_params_to_update(cifar_alexnet, feature_extract)
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    # 학습 
    cifar_alexnet, cifar_history = train(
        cifar_alexnet, cifar_dataloaders, criterion, 
        optimizer, device, num_epochs=num_epochs
    )
    
    # 모델 저장
    torch.save({
        'model_state_dict': cifar_alexnet.state_dict(),
        'history': cifar_history,
        'epoch': num_epochs,
        'best_acc': max(cifar_history['val_acc'])
    }, f'{save_dir}/cifar_alexnet_best.pth')
    print(f"CIFAR-10 model saved to {save_dir}/cifar_alexnet_best.pth")
    
    # 학습 결과 시각화/저장
    plot_training_history(
        cifar_history, 
        'CIFAR-10', 
        save_path=f'{results_dir}/cifar_training_history.png'
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()