import torch

def test(model, dataloaders, device, max_wrong_examples=5):

    model.eval()

    correct = 0
    total = 0

    wrong_examples = []

    with torch.no_grad():
        for data in dataloaders['test']:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if len(wrong_examples) < max_wrong_examples:
                wrong_mask = (predicted != labels)
                wrong_indices = torch.where(wrong_mask)[0]
                
                for i in wrong_indices:
                    if len(wrong_examples) >= max_wrong_examples:
                        break

                    wrong_examples.append({
                        'review' : input_ids[i].cpu(),
                        'predicted' : predicted[i].item(),
                        'actual': labels[i].item()
                    })

        accuracy = 100 * correct / total
        return accuracy, wrong_examples

        