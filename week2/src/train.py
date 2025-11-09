
import torch
import time
import copy
from tqdm import tqdm

# 참고: https://github.com/Prerna5194/IMDB_BERT_TRANSFORMER/blob/master/IMDB_MovieReviews_Classifier.ipynb

def train(model, dataloaders, criterion, optimizer, device, scheduler, num_epochs=10):
    '''
    model: transformers 모델

    '''
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("="*70)
    print("MODEL DEBUG INFO:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'token_embedding'):
        print(f"Vocab size: {model.encoder.token_embedding.num_embeddings}")
    print("="*70)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else: 
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(
                dataloaders[phase], 
                desc=f"{phase.upper():5s}",
                total=len(dataloaders[phase])
            )

            '''
                batch_idx가 없으면 에러가 남: enumerate(pbar)를 하면 (0, {'input_ids':...}) 이런 식으로 오기 때문에
                batch_idx를 넣거나 for data in pbar를 해야 함 ...

                enumerate(iterable)은 (index, value) 튜플을 반환
            '''
            for batch_idx, data in enumerate(pbar):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                scheduler.step()
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, {
            'train_acc': train_acc_history,
            'train_loss': train_loss_history,
            'val_acc': val_acc_history,
            'val_loss': val_loss_history
        }