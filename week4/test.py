import torch
from pathlib import Path
from tqdm import tqdm

from config import get_config, get_weights_file_path
from train import get_dataset, get_model, greedy_decode

def test_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Load dataset and tokenizers
    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    
    # Load model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Load trained weights
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
    else:
        # Load the latest model if preload not specified
        model_folder = Path(config["model_folder"])
        model_files = list(model_folder.glob("*.pt"))
        if not model_files:
            raise FileNotFoundError("No trained model found. Please train the model first.")
        model_filename = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading model: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    
    # Set model to evaluation mode
    model.eval()
    
    console_width = 80
    
    print("\n" + "="*console_width)
    print("Starting Model Evaluation")
    print("="*console_width + "\n")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for testing"
            
            # Perform greedy decoding
            model_out = greedy_decode(
                model, 
                encoder_input, 
                encoder_mask, 
                tokenizer_src, 
                tokenizer_tgt, 
                config['seq_len'], 
                device
            )
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            # Print results
            print('-'*console_width)
            print(f'Test Example {idx+1}:')
            print(f'Source ({config["lang_src"]}): {source_text}')
            print(f'Target ({config["lang_tgt"]}): {target_text}')
            print(f'Predicted: {model_out_text}')
            print('-'*console_width)
            print()
    
    print("\n" + "="*console_width)
    print("Testing Complete!")
    print("="*console_width)

def test_single_sentence(config, sentence):
    """Test a single sentence translation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # Load dataset and tokenizers
    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    
    # Load model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Load trained weights
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
    else:
        model_folder = Path(config["model_folder"])
        model_files = list(model_folder.glob("*.pt"))
        if not model_files:
            raise FileNotFoundError("No trained model found. Please train the model first.")
        model_filename = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading model: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    
    # Tokenize input sentence
    seq_len = config['seq_len']
    
    # Encode the source sentence
    src_tokens = tokenizer_src.encode(sentence).ids
    
    # Add SOS and EOS tokens
    num_padding_tokens = seq_len - len(src_tokens) - 2
    if num_padding_tokens < 0:
        raise ValueError(f"Sentence is too long. Max length: {seq_len - 2}")
    
    # Create encoder input
    encoder_input = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
        torch.tensor(src_tokens, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[PAD]')] * num_padding_tokens, dtype=torch.int64)
    ], dim=0).unsqueeze(0).to(device)
    
    # Create encoder mask
    encoder_mask = (encoder_input != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
    
    with torch.no_grad():
        # Perform greedy decoding
        model_out = greedy_decode(
            model,
            encoder_input,
            encoder_mask,
            tokenizer_src,
            tokenizer_tgt,
            seq_len,
            device
        )
        
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    
    print(f"\nSource ({config['lang_src']}): {sentence}")
    print(f"Translation ({config['lang_tgt']}): {model_out_text}")
    
    return model_out_text

if __name__ == '__main__':
    import sys
    
    config = get_config()
    
    # If a sentence is provided as command line argument, translate it
    if len(sys.argv) > 1:
        sentence = ' '.join(sys.argv[1:])
        test_single_sentence(config, sentence)
    else:
        # Otherwise, run full test on test dataset
        test_model(config)