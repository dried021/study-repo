import os
import torch
import pandas as pd

import numpy as np
from torch.utils.data import Dataset, DataLoader


# 참고: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
class IMDBDatasetBERT(Dataset):
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

        data_file = os.path.join(data_dir, f'imdb_{data_type}_BERT.npz')
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
    
class IMDBDatasetRNN(Dataset):
    def __init__(self, data_dir, data_type):
        self.data_dir = data_dir
        self.data_type = data_type

        data_file = os.path.join(data_dir, f'imdb_{data_type}_RNN.npz')
        data = np.load(data_file)

        self.input_ids = data['input_ids']
        self.labels = data['labels'].astype(np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.input_ids[index], dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }
    
def get_bert_dataloaders(data_dir, batch_size=32, device='cpu'):
    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())
    
    train_dataset = IMDBDatasetBERT(
        data_dir = data_dir,
        data_type='train'
    )

    val_dataset = IMDBDatasetBERT(
        data_dir = data_dir,
        data_type='val'
    )

    test_dataset = IMDBDatasetBERT(
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

def collate_fn_rnn(batch):
    """
    RNN/Transformer용 collate function - attention mask 생성
    
    보통 collate_fn은 패딩해줄 때 자주 사용됨
    """
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    
    # 패딩 적용
    input_ids_padded = pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=0  # <pad> token id
    )
    # batch_first가 true일 경우에는 B x T x * 형태로 만듬 (T는 제일 긴 sequence의 길이)
    
    # Attention mask 생성 (0이 아닌 곳이 1)
    attention_mask = (input_ids_padded != 0).long()
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask,
        'labels': labels
    }


def get_rnn_dataloaders(data_dir, batch_size=32, device='cpu'):
    """RNN용 DataLoader"""
    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())
    
    train_dataset = IMDBDatasetRNN(data_dir, 'train')
    val_dataset = IMDBDatasetRNN(data_dir, 'val')
    test_dataset = IMDBDatasetRNN(data_dir, 'test')

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            collate_fn=collate_fn_rnn
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            collate_fn=collate_fn_rnn
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin_memory,
            persistent_workers=True,
            collate_fn=collate_fn_rnn
        )
    }

    return dataloaders, test_dataset