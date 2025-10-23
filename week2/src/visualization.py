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
