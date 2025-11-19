import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from src.vocabulary import Tokenizer
from src.visualization import visualize_unk_ratio

from config import data_dir, vocab_size, results_dir
import re

def preprocess_text(text):
    # 괄호 안 모두 제거
    text = re.sub(r'\([^)]*\)', '', text)
    # text = re.sub(r'\((cnn|pictured)\)', '', text, flags=re.IGNORECASE)

    # 영문자/숫자/공백/문장부호 제외 제거
    text = re.sub(r'([.,!?])', r' \1 ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\d[\d,]*', '<num>', text)
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)

    # 앞 뒤 공백 제거
    text = text.strip()

    return text

def encode(reviews, tokenizer, batch_size=500, max_length=512):
    encoded_list = []
    token_lengths = []

    for text in tqdm(reviews, desc="Encoding... "):
        tokens = tokenizer.encode(text, max_length)
        tokens = tokens[:max_length]
        padding_length = max_length - len(tokens)
        tokens += [tokenizer.word2idx["<pad>"]] * padding_length

        encoded_list.append(tokens)
        token_lengths.append(len(tokens))

    import numpy as np
    print(f"평균: {np.mean(token_lengths)}")
    print(f"중간값: {np.median(token_lengths)}")
    print(f"95 percentile: {np.percentile(token_lengths, 95)}")
    print(f"최대: {np.max(token_lengths)}")
    print(f"512 초과: {sum(1 for l in token_lengths if l > 512)} / {len(token_lengths)}")

    return np.array(encoded_list, dtype=np.int64)

def preprocess_data(file_path, data_dir):
    
    phases = ['train', 'validation', 'test']
    data = {}

    for phase in phases:
        df = pd.read_csv(os.path.join(file_path, f'{phase}.csv'))
        df_sampled = df.sample(frac=0.1, random_state=42)
        
        articles = df_sampled['article'].tolist()
        highlights = df_sampled['highlights'].tolist()

        data[phase] = {}

        data[phase]['articles'] = [preprocess_text(article) for article in tqdm(articles)]
        data[phase]['highlights'] = [preprocess_text(highlight) for highlight in tqdm(highlights)]

    tokenizer = Tokenizer(vocab_size=vocab_size)

    vocab_builder = data['train']['articles']+data['train']['highlights']
    tokenizer.build_vocab(vocab_builder)
    tokenizer.save_vocab(os.path.join(data_dir, 'tokenizer.json'))

    visualize_unk_ratio(vocab_builder, tokenizer, max_length=512, save_path=f'{results_dir}/unk_ratio.png')

    data_arrays = {}

    for phase in phases:
        articles = encode(
            data[phase]['articles'],
            tokenizer,
            max_length=512
        )

        highlights = encode(
            data[phase]['highlights'],
            tokenizer,
            max_length=512
        )

        data_arrays[phase] = {
            'articles' : articles,
            'highlights' : highlights,
        }

    save_array(data_arrays, data_dir)

    return data_arrays

def save_array(arrays, data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)
    
    for split in ['train', 'validation', 'test']:
        print(f"Saving CNN/Daily Mail {split} data ...")
        
        save_dict = {
            'articles': arrays[split]['articles'],
            'highlights': arrays[split]['highlights']
        }
        
        np.savez_compressed(
            os.path.join(data_dir, f'cnn_{split}.npz'),
            **save_dict
        )

if __name__ == "__main__":
    data_arrays = preprocess_data('./data/cnn_dailymail', data_dir)

    print("\n=== Summary ===")
    print(f"Train: {data_arrays['train']['highlights'].shape}")
    print(f"Validation: {data_arrays['validation']['highlights'].shape}")
    print(f"Test: {data_arrays['test']['highlights'].shape}")


'''
=== UNK Token Analysis ===
Total tokens: 15,072,122
UNK tokens: 667,638
Overall UNK ratio: 4.43%

Per-review UNK ratio statistics:
  Mean: 4.67%
  Median: 4.12%
  Std: 3.22%
  Min: 0.00%
  Max: 36.36%
'''