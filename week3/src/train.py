import torch.optim as optim
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import os
import copy

from src.visualization import plot_training_history

class Trainer:
    def __init__(self, 
                model: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                device: torch.device,
                clip_grad_norm: float = 1.0,
                label_smoothing: float = 0.1,
                save_dir : str = "../saved_models",
                result_dir : str = "../results",
                save_every_n_epochs: int = 3  # 추가
                 ):
        
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.save_every_n_epochs = save_every_n_epochs  # 추가
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()

        total_loss =0.0
        pbar = tqdm(self.train_dataloader, desc="Training")
        
        for data in pbar:
            src = data['src'].to(self.device)
            tgt = data['tgt'].to(self.device)
            src_mask = data['src_mask'].to(self.device)

            tgt_seq_len = tgt.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len, self.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(src.size(0), 1, -1, -1)

            self.optimizer.zero_grad()
            output = self.model(src, tgt, src_mask, tgt_mask)
            loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            pbar.set_postfix({"loss" : loss.item()})

        return total_loss / len(self.train_dataloader)
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc="Validation")

            for data in pbar:
                src = data['src'].to(self.device)
                tgt = data['tgt'].to(self.device)
                src_mask = data['src_mask'].to(self.device)

                tgt_seq_len = tgt.size(1)
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len, self.device)
                tgt_mask = tgt_mask.unsqueeze(0).expand(src.size(0), 1, -1, -1)

                output = self.model(src, tgt, src_mask, tgt_mask)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)
    
    def train(self, num_epochs):
        print("="*70)
        print("MODEL DEBUG INFO:")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'token_embedding'):
            print(f"Vocab size: {self.model.encoder.token_embedding.num_embeddings}")
        print("="*70)

        val_loss_history = []
        train_loss_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = float('inf')

        since = time.time()

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch()
            train_loss_history.append(train_loss)

            val_loss = self.validate()
            val_loss_history.append(val_loss)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
            
            # Best model 업데이트
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
            # 3 epoch마다 체크포인트 저장
            if epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(
                    epoch, 
                    train_loss_history, 
                    val_loss_history,
                    is_best=(val_loss == best_val_loss)
                )
                print(f"💾 Checkpoint saved at epoch {epoch}")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_val_loss))

        # 최종 best model 저장
        self.model.load_state_dict(best_model_wts)
        history = {
            'train_loss' : train_loss_history,
            'val_loss' : val_loss_history
        }

        self.save_model(num_epochs, history)

    def save_checkpoint(self, epoch, train_loss_history, val_loss_history, is_best=False):
        """3 epoch마다 체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'current_train_loss': train_loss_history[-1],
            'current_val_loss': val_loss_history[-1]
        }
        
        # epoch별 체크포인트 저장
        checkpoint_path = f'{self.save_dir}/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 현재가 best model이면 별도 표시
        if is_best:
            best_checkpoint_path = f'{self.save_dir}/checkpoint_epoch_{epoch}_BEST.pth'
            torch.save(checkpoint, best_checkpoint_path)

    def save_model(self, num_epochs, history):
        """최종 best model 저장"""
        torch.save({
            'model_state_dict' : self.model.state_dict(),
            'history' : history,
            'epoch' : num_epochs,
            'best_loss' : min(history['val_loss'])
        }, f'{self.save_dir}/transformer_best.pth')
            
        plot_training_history(
        history, 
        "Transformer", 
        save_path=f'{self.result_dir}/training_history.png'
        )