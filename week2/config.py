import os
import torch

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, 'data')
save_dir = os.path.join(BASE_DIR, 'saved_models')
results_dir = os.path.join(BASE_DIR, 'results')

# data 설정
train_ratio = 0.8
val_ratio = 0.2
num_labels = 2

batch_size = 32

# 학습 설정
learning_rate = 0.001
momentum = 0.9
num_epochs = 5

# Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 레이블 설정
imdb_labels_map={
    0: 'positive',
    1: 'negative'
}