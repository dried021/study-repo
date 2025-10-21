import torch
import os

from config import *
from src.models import initialize_model
from src.test import test, predict_batch
from src.visualization import plot_wrong_predictions, plot_sample_images

from src.dataset import get_mnist_dataloaders, get_cifar_dataloaders

mnist_dataloaders, mnist_test = get_mnist_dataloaders(data_dir, batch_size)
cifar_dataloaders, cifar_test = get_cifar_dataloaders(data_dir, batch_size)

def main():
    print(f"Using device: {device}")
    print("-" * 50)
    
    # ==================== 모델 로드 ====================
    print("\n[1] Loading models...")
    
    # MNIST 모델 로드
    mnist_model, _ = initialize_model("alexnet", num_classes, feature_extract)
    mnist_model = mnist_model.to(device)
    mnist_checkpoint = torch.load(f'{save_dir}/mnist_alexnet_best.pth')
    mnist_model.load_state_dict(mnist_checkpoint['model_state_dict'])
    mnist_model.eval()
    
    print(f"MNIST Model Loaded")
    print(f"  - Best Accuracy: {mnist_checkpoint['best_acc']:.4f}")
    print(f"  - Trained Epochs: {mnist_checkpoint['epoch']}")
    
    # CIFAR-10 모델 로드
    cifar_model, _ = initialize_model("alexnet", num_classes, feature_extract)
    cifar_model = cifar_model.to(device)
    cifar_checkpoint = torch.load(f'{save_dir}/cifar_alexnet_best.pth')
    cifar_model.load_state_dict(cifar_checkpoint['model_state_dict'])
    cifar_model.eval()
    
    print(f"\nCIFAR-10 Model Loaded")
    print(f"  - Best Accuracy: {cifar_checkpoint['best_acc']:.4f}")
    print(f"  - Trained Epochs: {cifar_checkpoint['epoch']}")
    
    # ==================== 테스트 ====================
    print("\n[2] Testing models...")
    print("=" * 50)
    
    # MNIST 테스트
    mnist_accuracy, mnist_wrong = test(mnist_model, mnist_dataloaders, device)
    print(f"MNIST AlexNet Test Accuracy: {mnist_accuracy:.2f}%")
    
    # CIFAR-10 테스트
    cifar_accuracy, cifar_wrong = test(cifar_model, cifar_dataloaders, device)
    print(f"CIFAR-10 AlexNet Test Accuracy: {cifar_accuracy:.2f}%")
    
    # ==================== 샘플 예측 ====================
    print("\n[3] Sample predictions...")
    print("=" * 50)
    
    # MNIST 배치 예측
    mnist_pred, mnist_labels = predict_batch(
        mnist_model, mnist_dataloaders['test'], device
    )
    print(f"\nMNIST Predictions: {mnist_pred[:10]}")
    print(f"MNIST Actual:      {mnist_labels[:10]}")
    
    # CIFAR-10 배치 예측
    cifar_pred, cifar_labels = predict_batch(
        cifar_model, cifar_dataloaders['test'], device
    )
    
    pred_names = [cifar_labels_map[p.item()] for p in cifar_pred[:10]]
    actual_names = [cifar_labels_map[l.item()] for l in cifar_labels[:10]]
    
    print(f"\nCIFAR-10 Predictions: {pred_names}")
    print(f"CIFAR-10 Actual:      {actual_names}")
    
    # ==================== 시각화 ====================
    print("\n[4] Visualizing results...")
    print("=" * 50)
    
    # 학습 히스토리 시각화 (이미 저장된 것 로드)
    from src.visualization import plot_training_history
    plot_training_history(
        mnist_checkpoint['history'], 
        'MNIST',
        save_path=f'{results_dir}/mnist_test_history.png'
    )
    plot_training_history(
        cifar_checkpoint['history'], 
        'CIFAR-10',
        save_path=f'{results_dir}/cifar_test_history.png'
    )
    
    # 오답 시각화
    plot_wrong_predictions(
        mnist_wrong, 
        dataset_name='MNIST',
        save_path=f'{results_dir}/mnist_wrong_predictions.png'
    )
    
    plot_wrong_predictions(
        cifar_wrong, 
        labels_map=cifar_labels_map,
        dataset_name='CIFAR-10',
        save_path=f'{results_dir}/cifar_wrong_predictions.png'
    )
    
    # 샘플 이미지 시각화 (mnist_test, cifar_test는 미리 정의되어 있어야 함)
    plot_sample_images(
        mnist_test, 
        cifar_test, 
        cifar_labels_map,
        save_path=f'{results_dir}/sample_images.png'
    )
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    print(f"Results saved to {results_dir}/")
    print("=" * 50)

if __name__ == "__main__":
    main()