import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import BartModel, LongformerForSequenceClassification, BertModel
from src.visualization import plot_training_history
from src.model import initialize_model

from config import device, save_dir, results_dir, data_dir, batch_size, num_labels, learning_rate, momentum, num_epochs

from src.train import train
from src.dataset import get_imdb_dataloaders

os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def main(model_type):
    print(f"Using device: {device}")
    print("-"*50)

    imdb_dataloaders, test_data = get_imdb_dataloaders(data_dir, batch_size, device=str(device))

    model = initialize_model(model_type, num_labels)
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    model_trained, history = train(
        model, imdb_dataloaders, criterion,
        optimizer, device, num_epochs = num_epochs
    )

    torch.save({
        'model_state_dict' : model_trained.state_dict(),
        'history' : history,
        'epoch' : num_epochs,
        'best_acc' : max(history['val_acc'])
    }, f'{save_dir}/{model_type}_best.pth')

    plot_training_history(
        history, 
        'MNIST', 
        save_path=f'{results_dir}/{model_type}_training_history.png'
    )

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Choose Model: Longformer, BART')
    parser.add_argument('--model', type=str, default = 'bert')
    args = parser.parse_args()

    main(args.model)
