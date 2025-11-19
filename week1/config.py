import torch
import os

#Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 실행하는 훈련 epoch 수
num_epochs = 10

# 훈련에 사용되는 배치 크기
batch_size = 32

learning_rate = 0.001

momentum = 0.9

# 데이터셋의 class 수
num_classes = 10 #MNIST / CIFAR-10


#모델 설정
feature_extract = True  # fine tuning or feature extraction
use_pretrained = True   # pretrained 사용 여부


#경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(BASE_DIR, 'data')
save_dir = os.path.join(BASE_DIR, 'saved_models')
results_dir = os.path.join(BASE_DIR, 'results')

#CIFAR_10 레이블
cifar_labels_map={
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}