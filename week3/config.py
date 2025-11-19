import os
import torch

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, 'data')
save_dir = os.path.join(BASE_DIR, 'saved_models')
results_dir = os.path.join(BASE_DIR, 'results')

# data 설정
batch_size = 16
vocab_size = 20000 

# 학습 설정
learning_rate = 3e-4
num_epochs = 15

clip_grad_norm = 1.0
label_smoothing = 0.1

# Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_config = {
    'vocab_size' : 20000,
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'd_ff': 1024,
    'max_len': 512,
    'dropout': 0.1
}
