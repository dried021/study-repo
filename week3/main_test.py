import torch
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.transformer import Transformer, TransformerEncoder, TransformerDecoder
from src.dataset import get_dataloaders
from src.vocabulary import Tokenizer
from config import *

from src.visualization import plot_training_history


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    
    encoder = TransformerEncoder(**model_config)
    decoder = TransformerDecoder(**model_config)
    model = Transformer(encoder, decoder)
    
    # state_dict 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    best_loss = checkpoint.get('best_loss', 'N/A')
    print(f"Best validation loss: {best_loss if isinstance(best_loss, str) else f'{best_loss:.4f}'}")
    return model

def calculate_bleu(references, hypotheses):
    from sacrebleu import corpus_bleu
    return corpus_bleu([[ref.split()] for ref in references], 
                       [hyp.split() for hyp in hypotheses])

def generate_summary(model, src, src_mask, tokenizer, device, max_len=100):
    """요약 생성 (<unk> 페널티 추가)"""
    model.eval()
    
    sos_idx = tokenizer.word2idx['<sos>']
    eos_idx = tokenizer.word2idx['<eos>']
    unk_idx = tokenizer.word2idx['<unk>']  # 추가
    
    tgt = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_seq_len = tgt.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len, device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(1, 1, -1, -1)
            
            output = model(src, tgt, src_mask, tgt_mask)
            logits = output[0, -1, :]
            
            # <unk> 토큰에 페널티 부여
            logits[unk_idx] -= 5.0  # 큰 페널티
            
            next_token = logits.argmax().item()
            
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
            
            if next_token == eos_idx:
                break
    
    return tgt

def test_model(model, dataloader, tokenizer, device):
    """Test model on test dataset."""
    model.eval()
    
    all_references = []
    all_hypotheses = []
    
    print("\n=== Testing Model ===")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device)
            
            # Generate summaries
            batch_size = src.size(0)
            for j in range(batch_size):
                # Generate summary
                generated = generate_summary(
                    model, 
                    src[j:j+1], 
                    src_mask[j:j+1], 
                    tokenizer, 
                    device
                )
                
                # Decode
                reference = tokenizer.decode(tgt[j].cpu().tolist())
                hypothesis = tokenizer.decode(generated[0].cpu().tolist())
                
                all_references.append(reference)
                all_hypotheses.append(hypothesis)
                
                # Print first 5 examples
                if i == 0 and j < 5:
                    source = tokenizer.decode(src[j].cpu().tolist())
                    print(f"\n[Example {j+1}]")
                    print(f"Source: {source}")
                    print(f"Reference: {reference}")
                    print(f"Generated: {hypothesis}")
                    print("-" * 80)
    
    # Calculate BLEU score
    try:
        bleu_score = calculate_bleu(all_references, all_hypotheses)
        print(f"\nBLEU Score: {bleu_score:.4f}")
    except:
        print("\nCould not calculate BLEU score (install nltk)")
    
    return all_references, all_hypotheses

def interactive_summarization(model, tokenizer, device):
    """Interactive mode for summarization."""
    print("\n=== Interactive Summarization Mode ===")
    print("Enter text to summarize (type 'quit' to exit)")
    
    while True:
        text = input("\nText to summarize: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        # Encode text
        tokens = tokenizer.encode(text)
        src = torch.tensor([tokens], dtype=torch.long).to(device)
        src_mask = (src != tokenizer.word2idx['<pad>']).unsqueeze(1).unsqueeze(2)
        
        # Generate summary
        generated = generate_summary(model, src, src_mask, tokenizer, device)
        summary = tokenizer.decode(generated[0].cpu().tolist())
        
        print(f"Summary: {summary}")

def main():
    parser = argparse.ArgumentParser(description='Test Transformer Summarization Model')
    parser.add_argument('--model_path', type=str, default='./saved_models/transformer_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=data_dir,
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive mode')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("-" * 50)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path, device)
    
    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_vocab(os.path.join(args.data_dir, 'tokenizer.json'))
    
    # Get test dataloader
    print("\nLoading test data...")
    dataloaders = get_dataloaders(args.data_dir, args.batch_size, device=str(device))
    test_dataloader = dataloaders['test']
    
    # Test model
    references, hypotheses = test_model(model, test_dataloader, tokenizer, device)
    
    # Save results
    output_file = os.path.join(results_dir, 'test_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for ref, hyp in zip(references, hypotheses):
            f.write(f"Reference: {ref}\n")
            f.write(f"Generated: {hyp}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\nResults saved to {output_file}")
    
    # Interactive mode
    if args.interactive:
        interactive_summarization(model, tokenizer, device)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()
