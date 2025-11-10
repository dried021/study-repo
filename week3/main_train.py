import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

from torch.optim import AdamW
from src.visualization import plot_training_history
from src.transformer import Transformer, TransformerEncoder, TransformerDecoder

from config import *

from src.train import Trainer
from src.dataset import get_dataloaders

os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def main():
    print(f"Using device: {device}")
    print("-"*50)

    dataloaders = get_dataloaders(data_dir, batch_size, device=str(device))

    
    encoder = TransformerEncoder(**model_config)
    decoder = TransformerDecoder(**model_config)
    model = Transformer(encoder, decoder)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CosineAnnealingLR(optimizer, T_max = len(dataloaders['train']) * num_epochs)

    trainer = Trainer(model, dataloaders['train'], dataloaders['val'], optimizer, scheduler, device,
                      clip_grad_norm, label_smoothing, save_dir, results_dir)

    trainer.train(num_epochs=num_epochs)

if __name__ == "__main__":
    main()
