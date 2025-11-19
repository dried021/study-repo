import os
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data_dir, data_type, max_len = 512):
        self.data_dir = data_dir
        self.max_len = max_len
        
        data_file = os.path.join(data_dir, f'cnn_{data_type}.npz')
        data = np.load(data_file)

        self.articles = data['articles']
        self.highlights = data['highlights']

    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, index):
        return {
            'articles': torch.tensor(self.articles[index], dtype=torch.long),
            'highlights': torch.tensor(self.highlights[index], dtype=torch.long),
        }
    
def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    articles = [item['articles'] for item in batch]
    highlights = [item['highlights'] for item in batch]

    articles_padded = pad_sequence(
        articles,
        batch_first=True,
        padding_value=0
    )

    highlights_padded = pad_sequence(
        highlights,
        batch_first=True,
        padding_value=0
    )

    source_mask = (articles_padded != 0).unsqueeze(1).unsqueeze(2)
    
    return {
        'src' : articles_padded,
        'tgt' : highlights_padded,
        'src_mask' : source_mask
    }

def get_dataloaders(data_dir, batch_size = 32, device='cpu'):
    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())

    train_dataset = TextDataset(data_dir, 'train')
    validation_dataset = TextDataset(data_dir, 'validation')
    test_dataset = TextDataset(data_dir, 'test')

    dataloaders = {
        'train' : DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            collate_fn=collate_fn
        ),
        'val' : DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            collate_fn=collate_fn
        ),
        'test' : DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            collate_fn=collate_fn
        )
    }

    return dataloaders