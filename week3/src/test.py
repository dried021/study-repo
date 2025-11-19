import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def calculate_perplexity(self, dataloader) -> float:
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Calculating Perplexity")
            for data in pbar:
                src = data['src'].to(self.device)
                tgt = data['tgt'].to(self.device)
                src_mask = data['src_mask'].to(self.device)

                batch_size = src.size(0)
                tgt_seq_len = tgt.size(1)
                
                # 마스크 생성
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len, self.device)
                tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, 1, -1, -1)

                # forward pass
                output = self.model(src, tgt, src_mask, tgt_mask)
                loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))

                total_loss += loss.item()
                # padding이 아닌 토큰 개수만 카운트
                total_tokens += (tgt != 0).sum().item()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity

    def summarize(
            self,
            text: str,
            max_len: int = 100,
            method: str = "greedy"  # "greedy" or "sampling"
    ) -> str:
        """
        텍스트 요약 생성
        src: 원본 텍스트, tgt: 요약문
        """
        # 원본 텍스트를 src로 인코딩
        src_tokens = torch.tensor(
            [self.tokenizer.encode(text)], 
            dtype=torch.long
        ).to(self.device)
        
        # src_mask 생성 (padding mask)
        src_mask = (src_tokens != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        
        # Encoder 통과
        with torch.no_grad():
            enc_output = self.model.encoder(src_tokens, src_mask)
        
        # Decoder를 위한 시작 토큰 (<sos> 가정)
        if "<sos>" in self.tokenizer.word2idx:
            tgt_tokens = torch.tensor(
                [[self.tokenizer.word2idx["<sos>"]]], 
                dtype=torch.long
            ).to(self.device)
        else:
            # <sos>가 없으면 첫 번째 토큰으로 시작
            tgt_tokens = torch.tensor([[1]], dtype=torch.long).to(self.device)

        # Autoregressive 요약 생성
        for _ in range(max_len):
            tgt_seq_len = tgt_tokens.size(1)
            
            # tgt_mask 생성 (causal mask)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len, self.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(1, 1, -1, -1)

            with torch.no_grad():
                # Decoder 통과
                output = self.model.decoder(tgt_tokens, enc_output, src_mask, tgt_mask)

            # 마지막 토큰의 logits
            logits = output[0, -1, :]

            if method == "greedy":
                # Greedy: 가장 높은 확률의 토큰 선택
                next_token = logits.argmax().item()
            else:
                # Sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            # 생성된 토큰 추가
            tgt_tokens = torch.cat(
                [tgt_tokens, torch.tensor([[next_token]], device=self.device)], 
                dim=1
            )

            # <eos> 토큰이면 종료
            if "<eos>" in self.tokenizer.word2idx:
                if next_token == self.tokenizer.word2idx["<eos>"]:
                    break
        
        # 디코딩 (요약문)
        summary_tokens = tgt_tokens[0].tolist()
        return self.tokenizer.decode(summary_tokens)

    def summarize_with_sampling(
            self,
            text: str,
            max_len: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.95
    ) -> str:
        """
        Sampling을 사용한 요약 생성 (더 다양한 결과)
        """
        # 원본 텍스트를 src로 인코딩
        src_tokens = torch.tensor(
            [self.tokenizer.encode(text)], 
            dtype=torch.long
        ).to(self.device)
        
        # src_mask 생성
        src_mask = (src_tokens != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        
        # Encoder 통과
        with torch.no_grad():
            enc_output = self.model.encoder(src_tokens, src_mask)
        
        # 시작 토큰
        if "<sos>" in self.tokenizer.word2idx:
            tgt_tokens = torch.tensor(
                [[self.tokenizer.word2idx["<sos>"]]], 
                dtype=torch.long
            ).to(self.device)
        else:
            tgt_tokens = torch.tensor([[1]], dtype=torch.long).to(self.device)

        # Autoregressive 생성
        for _ in range(max_len):
            tgt_seq_len = tgt_tokens.size(1)
            
            # tgt_mask 생성
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len, self.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(1, 1, -1, -1)

            with torch.no_grad():
                output = self.model.decoder(tgt_tokens, enc_output, src_mask, tgt_mask)

            # 마지막 토큰의 logits
            logits = output[0, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[-1]] = -float('inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('inf')
            
            # Sampling
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            tgt_tokens = torch.cat(
                [tgt_tokens, torch.tensor([[next_token]], device=self.device)], 
                dim=1
            )

            # <eos> 토큰이면 종료
            if "<eos>" in self.tokenizer.word2idx:
                if next_token == self.tokenizer.word2idx["<eos>"]:
                    break
        
        summary_tokens = tgt_tokens[0].tolist()
        return self.tokenizer.decode(summary_tokens)

    def evaluate_rouge(self, dataloader, max_samples=None):
        """
        ROUGE 스코어 계산 (요약 평가)
        rouge 라이브러리 필요: pip install rouge-score
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            print("rouge-score 라이브러리가 필요합니다: pip install rouge-score")
            return None
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating ROUGE")
            for idx, data in enumerate(pbar):
                if max_samples and idx >= max_samples:
                    break
                
                src = data['src'].to(self.device)
                tgt = data['tgt'].to(self.device)
                src_mask = data['src_mask'].to(self.device)
                
                batch_size = src.size(0)
                
                for i in range(batch_size):
                    # 원본 텍스트
                    src_text = self.tokenizer.decode(src[i].tolist())
                    # 정답 요약문
                    ref_summary = self.tokenizer.decode(tgt[i].tolist())
                    # 생성된 요약문
                    pred_summary = self.summarize(src_text, max_len=100, method="greedy")
                    
                    # ROUGE 스코어 계산
                    scores = scorer.score(ref_summary, pred_summary)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores),
            'rouge2': sum(rouge2_scores) / len(rouge2_scores),
            'rougeL': sum(rougeL_scores) / len(rougeL_scores)
        }