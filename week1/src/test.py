import torch

# test data로 모델을 평가하는 함수
def test(model, dataloaders, device, max_wrong_examples=5):

  model.eval()
  # 모델을 평가 모드로 전환

  correct = 0
  total = 0

  wrong_examples = []

  with torch.no_grad():
    # gradient 연산의 옵션을 끌 때 사용
    # requires_grad=False 상태가 되어 메모리 사용량을 아껴줌

    for data in dataloaders['test']:
    
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # tensor 연산을 위해 device로 옮김

        # train에서처럼 행에서 제일 큰 값을 prediction으로 설정
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # 한 배치에서의 데이터 개수를 저장
        total += labels.size(0)
        # 텐서값을 모두 더하고 scalar로 변환
        correct += (predicted == labels).sum().item()

        # 오류 사례를 다 수집하지 않았을 경우
        if len(wrong_examples) < max_wrong_examples:
            for i in range(len(labels)):
                # len(labels) = 이 배치의 data 개수

                # 이 배치에서 오류를 다 수집했으면 오류 사례 수집 종료
                if len(wrong_examples) >= max_wrong_examples:
                    break 

                # 오류 사례 저장    
                if predicted[i] != labels[i]:
                    wrong_examples.append({
                        'image': images[i].cpu(),
                        'predicted': predicted[i].item(),
                        'actual': labels[i].item()
                    })

  accuracy = 100*correct / total
  return accuracy, wrong_examples


# 첫번째 배치에서 predicted 값을 뽑아옴
def predict_batch(model, dataloader, device):
    model.eval()
    
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    return predicted, labels