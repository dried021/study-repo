import torch

#Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 10
batch_size = 32
learning_rate = 0.001
momentum = 0.9
num_classes = 10 #MNIST / CIFAR-10


#모델 설정
feature_extract = True
use_pretrained = True


#경로 설정
data_dir = "./data"
save_dir = "./saved_models"
results_dir="./results"

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