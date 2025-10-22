import torch
import time
import copy
# https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html

# train: 주어진 모델의 학습과 검증을 처리
def train(model, dataloaders, criterion, optimizer, device, num_epochs=25):
  '''
  train args
  
  model: pytorch 모델
  dataloaders: 데이터로더 딕셔너리
  criterion: 손실 함수 torch.nn
  optimizer: 최적화 torch.optim
  device: cpu / gpu 여부
  num_epochs: 훈련 epochs 수
  '''
  since = time.time()

  # history 기록: train/val의 정확도/loss
  val_acc_history = []
  val_loss_history = []
  train_acc_history = []
  train_loss_history = []

  # state_dict(): model의 각 계층의 매개변수 텐서로 매핑되는 python 사전(dict) 객체
  #               활용해서 model을 저장/업데이트/변경/복원할 수 있음
  # 마지막 모델이 아닌 제일 좋은 정확도를 가진 모델을 최종적으로 선택하기 위해 parameter를 저장시켜 놓음
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-'*20)

    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
        # 모델을 학습하기 위해 호출되는 함수.
        # 모델의 파라미터 업데이트 / gradient 계산을 가능하게 함
        # 여러 정규화 기법(dropout/batch normalization 등)이 동작하도록 함
      else:
        model.eval()
        # 모델을 평가 모드로 전환
        # regularization 비활성 (batch normalization에서 이동 평균과 이동 분산이 업데이트 되지 않음)

      # 이 epoch에서 loss / corrects 초기화
      running_loss = 0.0
      running_corrects = 0

      # dataloaders는 반복 가능한 객체
      for inputs, labels in dataloaders[phase]:
        # 텐서 연산 시 모든 텐서가 동일한 operation(전부 cpu or 전부 gpu)를 이용해야 함
        # 데이터 텐서를 이동시킴
        inputs = inputs.to(device)
        labels = labels.to(device)

        # parameter 경사를 0으로 설정 (초기화)
        optimizer.zero_grad()

        #forward
        with torch.set_grad_enabled(phase=='train'):
          # 인수에 따라 grads를 활성화하거나 비활성화 
          # 인수가 True면 활성화 False면 비활성화
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          # prediction 값
          # torch.max하면 최댓값, index값을 tensor로 리턴
          # dim=1이므로 각 행마다 최댓값의 위치를 예측값으로 사용하겠다
          _, preds = torch.max(outputs, 1)

          if phase == 'train':
            loss.backward() # tensor에 대한 자동 미분. Autograd가 각 모델 매개변수의 기울기를 계산하여 매개변수의 속성에 저장
            optimizer.step() # 경사 하강법 시작. optimizer는 저장된 경사도를 사용하여 각 매개변수를 조정

        # loss.item()을 통해 tensor -> scalar 변환
        # 이 때 loss.item은 이 배치에서의 loss를 나타냄. inputs.size(0) (배치에서 data의 개수)를 곱해서 복원
        # 이 배치에서 예측값 = label인 개수를 세서 반환
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      # 이 epoch에서 loss는 running_loss/전체 데이터의 개수로 나눈 값 (전체 평균을 구해야 하므로)
      epoch_loss = running_loss/len(dataloaders[phase].dataset)
      epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      if phase == 'train':
        # train일 경우 epoch_loss와 acc 저장
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
      else:
        # val일 경우 epoch_loss와 acc 저장
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc)

        # val phase에서 best acc인 상태일 경우 저장
        if epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model_wts = copy.deepcopy(model.state_dict())

      print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # 제일 좋은 acc의 parameter를 불러옴
  model.load_state_dict(best_model_wts)

  # 훈련된 모델과 history를 반환. history는 dictionary 형태
  return model, {
        'train_acc': train_acc_history,
        'train_loss': train_loss_history,
        'val_acc': val_acc_history,
        'val_loss': val_loss_history
    }

