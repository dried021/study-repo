import matplotlib.pyplot as plt
import os

def plot_training_history(history, dataset_name, save_path=None):
    train_acc = [acc.cpu().item() for acc in history['train_acc']]
    val_acc = [acc.cpu().item() for acc in history['val_acc']]
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = range(1, len(train_acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

    # Accuracy
    ax1.plot(epochs, train_acc, 'b-o', label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, val_acc, 'r-s', label="Val Accuracy", linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{dataset_name} Training and Validation Accuracy',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'{dataset_name} Training and Validation Loss', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


## 오답사례 시각화
def plot_wrong_predictions(wrong_examples, labels_map=None, dataset_name='', 
                          save_path=None):
    n_examples = len(wrong_examples)
    
    figure = plt.figure(figsize=(15, 3))
    
    for i in range(n_examples):
        image = wrong_examples[i]['image'] * 1.0 + 0.5
        image = image.permute(1, 2, 0)
        
        figure.add_subplot(1, n_examples, i + 1)
        
        pred = wrong_examples[i]['predicted']
        actual = wrong_examples[i]['actual']
        
        if labels_map:
            pred_label = labels_map[pred]
            actual_label = labels_map[actual]
            title = f"Pred: {pred_label}\nTrue: {actual_label}"
        else:
            title = f"Pred: {pred}\nTrue: {actual}"
        
        plt.title(title, color='red')
        plt.axis('off')
        plt.imshow(image)
    
    plt.suptitle(f'{dataset_name} Wrong Predictions', fontsize=16, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


## 샘플 이미지 시각화
def plot_sample_images(mnist_dataset, cifar_dataset, cifar_labels_map, 
                      n_samples=4, save_path=None):
    figure = plt.figure(figsize=(12, 6))
    
    # MNIST
    for i in range(n_samples):
        image, label = mnist_dataset.get_item_for_viz(i)
        image = image.permute(1, 2, 0)
        figure.add_subplot(2, n_samples, i + 1)
        plt.title(f"MNIST: {label}")
        plt.axis('off')
        plt.imshow(image)
    
    # CIFAR-10
    for i in range(n_samples):
        image, label = cifar_dataset.get_item_for_viz(i)
        image = image.permute(1, 2, 0)
        figure.add_subplot(2, n_samples, i + n_samples + 1)
        label_name = cifar_labels_map[label]
        plt.title(f"CIFAR: {label_name}")
        plt.axis('off')
        plt.imshow(image)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()