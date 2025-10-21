import torch

def test(model, dataloaders, device, max_wrong_examples=5):
  model.eval()

  correct = 0
  total = 0

  wrong_examples = []

  with torch.no_grad():
    for data in dataloaders['test']:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      for i in range(len(labels)):
          if len(wrong_examples) >= 5:
              break
          if predicted[i] != labels[i]:
              wrong_examples.append({
                  'image': images[i].cpu(),
                  'predicted': predicted[i].item(),
                  'actual': labels[i].item()
              })

      if len(wrong_examples) >= max_wrong_examples:
          break

  accuracy = 100*correct / total
  return accuracy, wrong_examples