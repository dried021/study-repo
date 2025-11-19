import torch
from pathlib import Path
from tqdm import tqdm
import sacrebleu

from config import get_config, get_weights_file_path
from train import get_dataset, get_model, greedy_decode

def detokenize(text):
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    text = text.replace(' ;', ';')
    text = text.replace(' :', ':')
    return text

def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    train_dataloader,validation_dataloader, tokenizer_src, tokenizer_tgt, test_dataloader = get_dataset(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model_filename = get_weights_file_path(config) 
    
    print(f"Loading model: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    model.eval()

    console_width = 80
    
    print("\n" + "="*console_width)
    print("Starting Model Evaluation with BLEU Score")
    print("="*console_width + "\n")

    reference_texts = []
    predicted_texts = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for testing"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            predicted_texts.append(model_out_text)
            reference_texts.append(target_text)

    print("\nCalculating BLEU Score...")

    cleaned_predictions = [detokenize(text) for text in predicted_texts]
    cleaned_references = [detokenize(text) for text in reference_texts]

    bleu = sacrebleu.corpus_bleu(cleaned_predictions, [cleaned_references])

    print(f"\n{'='*console_width}")
    print(f"Corpus BLEU Score: {bleu.score:.2f}")
    print(f"{'='*console_width}")


if __name__ == '__main__':
    config = get_config()
    test_model(config)

