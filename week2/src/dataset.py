import os
import torch
import pandas as pd

import numpy as np
from torch.utils.data import Dataset, DataLoader


# 참고: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
class IMDBDataset(Dataset):
    """
    IMDB dataset
    """
    def __init__(self, data_dir, data_type):
        """
        data_dir: 데이터 파일 경로
        data_type(String): 'train'/'val'/'test'
        """

        self.data_dir = data_dir
        self.data_type = data_type

        if data_type == 'train':
            data_file = os.path.join(data_dir, 'imdb_train.npz')
        elif data_type == 'val':
            data_file = os.path.join(data_dir, 'imdb_val.npz')
        else:
            data_file = os.path.join(data_dir, 'imdb_test.npz')

        data = np.load(data_file)

        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels'].astype(np.int64)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.input_ids[index], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[index], dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }
    
def get_imdb_dataloaders(data_dir, batch_size=32, device='cpu'):
    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())
    
    train_dataset = IMDBDataset(
        data_dir = data_dir,
        data_type='train'
    )

    val_dataset = IMDBDataset(
        data_dir = data_dir,
        data_type='val'
    )

    test_dataset = IMDBDataset(
        data_dir = data_dir,
        data_type='test'
    )

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
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
