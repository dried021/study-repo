import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

from torch.optim import AdamW
from src.visualization import plot_training_history
from src.model import initialize_model

from config import device, save_dir, results_dir, data_dir, batch_size, num_labels, learning_rate, momentum, num_epochs, encoder_params, vocab_size

from src.train import train
from src.dataset import get_bert_dataloaders, get_rnn_dataloaders

os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def main(model_type):
    print(f"Using device: {device}")
    print("-"*50)

    if model_type == "bert":
        imdb_dataloaders, test_data = get_bert_dataloaders(data_dir, batch_size, device=str(device))
    elif model_type == "transformer":
        imdb_dataloaders, test_data = get_rnn_dataloaders(data_dir, batch_size, device=str(device)) 

    model = initialize_model(model_type, num_labels, vocab_size, encoder_params)
    model = model.to(device)

    total_steps = len(imdb_dataloaders['train']) * num_epochs
    warmup_steps = int(0.05 * total_steps)
    
    if model_type == "bert":
        optimizer = AdamW(
                model.parameters(),
                lr=2e-5,
                eps=1e-8,
                weight_decay=0.01
            )
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

        criterion = nn.CrossEntropyLoss().to(device)

    elif model_type == "transformer":
        optimizer = AdamW(
            model.parameters(),
            lr = 1e-4,
            betas = (0.9, 0.98),
            eps = 1e-9)

        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        criterion = nn.CrossEntropyLoss()


    model_trained, history = train(
        model, imdb_dataloaders, criterion,
        optimizer, device, scheduler, num_epochs = num_epochs
    )

    torch.save({
        'model_state_dict' : model_trained.state_dict(),
        'history' : history,
        'epoch' : num_epochs,
        'best_acc' : max(history['val_acc'])
    }, f'{save_dir}/{model_type}_best.pth')

    plot_training_history(
        history, 
        model_type.upper(), 
        save_path=f'{results_dir}/{model_type}_training_history.png'
    )

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Choose Model: transformer, bert')
    parser.add_argument('--model', type=str, default = 'bert')
    args = parser.parse_args()

    main(args.model)
