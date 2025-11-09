import torch
import os
import json

from config import *
from src.model import initialize_model
from src.test import test
from src.model import initialize_model
from src.dataset import get_bert_dataloaders, get_rnn_dataloaders
from config import device, save_dir, results_dir, data_dir, batch_size, num_labels, learning_rate, momentum, num_epochs, vocab_size, encoder_params
from transformers import BertTokenizer

def main(model_type):
    print(f"Using device: {device}")
    print("-" * 50)

    if model_type == "bert":
        dataloaders, test_data = get_bert_dataloaders(data_dir, batch_size, device=str(device))
        tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        #transformer
        dataloaders, test_data = get_rnn_dataloaders(data_dir, batch_size, device=str(device))


    model = initialize_model(model_type, num_labels, vocab_size, encoder_params)
    model = model.to(device)

    model_checkpoint = torch.load(f'{save_dir}/{model_type}_best.pth')
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()

    print(f"{model_type.upper()} Model Loaded")
    print(f"  - Best Accuracy: {model_checkpoint['best_acc']:.4f}")
    print(f"  - Trained Epochs: {model_checkpoint['epoch']}")

    # ==================== 테스트 ====================
    print("\n[2] Testing models...")
    print("=" * 50)

    bert_accuracy, bert_wrong = test(model, dataloaders, device)
    print(f"IMDB {model_type.upper()} Test Accuracy: {bert_accuracy:.2f}%")

    with open('./data/tokenizer_rnn.json', 'r') as f:
        vocab = json.load(f)
        idx2word = {int(k): v for k, v in vocab["idx2word"].items()}

    for wrong in bert_wrong:
        input_ids_tensor = wrong['review'].cpu() if wrong['review'].device != 'cpu' else wrong['review']
        if model_type=="bert":
            review_text = tokenizer_bert.decode(input_ids_tensor, skip_special_tokens=True)
        else:
            tokens = input_ids_tensor.tolist()
            review_text = ' '.join([idx2word.get(token, '<unk>') for token in tokens if token not in [0, 2, 3]])
            # 0=<pad>, 2=<sos>, 3=<eos> 제외
        
        print("Review:", review_text)
        print("Predicted:", wrong['predicted'], "Actual:", wrong['actual'])
        print("-" * 50)

    print("Testing completed!")
    print("=" * 50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Choose Model: transformer, bert')
    parser.add_argument('--model', type=str, default = 'bert')
    args = parser.parse_args()

    main(args.model)