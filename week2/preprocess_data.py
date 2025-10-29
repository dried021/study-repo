import pandas as pd
from transformers import BertTokenizer, LongformerTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os

from src.vocabulary import RNNTokenizer

from config import data_dir, train_ratio, val_ratio, vocab_size
import re



# 개별 텍스트 전처리 함수
def preprocess_text(text):
    # 파이썬 re, regex는 정규표현식
    
    # HTML 태그 제거
    text = re.sub(r'<br\s*/?\s*', ' ', text) # <br> 태그를 공백으로 변환
    text = re.sub(r'<.*?>', '', text) # 일반 HTML 태그 제거

    # 영문자/숫자/공백/문장부호 제외 제거
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)

    # 앞 뒤 공백 제거
    text = text.strip()

    return text

def split_reviews(reviews_cleaned, labels, train_ratio=0.8, val_ratio=0.2):
    '''
    전처리된 reviews를 train/val/test로 분할
    '''
    # train / test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        reviews_cleaned,
        labels,
        test_size=(1 - train_ratio),
        random_state=42,
        stratify=labels
    )

    # train / val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    return {
        'train': {'reviews': X_train, 'labels': y_train},
        'val': {'reviews': X_val, 'labels': y_val},
        'test': {'reviews': X_test, 'labels': y_test}
    }

def encode_split(reviews, tokenizer, batch_size=500, max_length=512):
    '''
    split된 reviews를 tokenization
    '''
    encoded_list = []

    for i in tqdm(range(0, len(reviews), batch_size), desc="Encoding"):
        batch = reviews[i: i + batch_size]
        encoded_batch = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=max_length,  # 512로 했을 때 한 epoch에 3시간이 걸려 256으로 진행 
            return_tensors='np'
        )
        encoded_list.append(encoded_batch)

    input_ids = np.concatenate([e['input_ids'] for e in encoded_list])
    attention_mask = np.concatenate([e['attention_mask'] for e in encoded_list])

    return input_ids, attention_mask


def encode_split_rnn(reviews, tokenizer_rnn, max_length=512):
    '''
    split된 reviews를 RNN tokenization
    '''
    all_tokens = []
    
    for review in tqdm(reviews, desc="Encoding (RNN)"):
        # 각 리뷰를 개별적으로 인코딩
        tokens = tokenizer_rnn.encode(review, max_length)
        
        # Padding
        tokens = tokens[:max_length]
        padding_length = max_length - len(tokens)
        tokens += [tokenizer_rnn.word2idx["<pad>"]] * padding_length
        
        all_tokens.append(tokens)
    
    return np.array(all_tokens, dtype=np.int64)
    
def preprocess_data(file_path, data_dir):
    '''
    csv 파일 위치를 input으로 받아
    인코딩된 inputs_ids, attention_mask, labels 반환
    '''
    df = pd.read_csv(file_path)
    
    reviews = df['review'].tolist()
    
    label_mapping = {'positive': 1, 'negative': 0}
    labels = df['sentiment'].map(label_mapping).to_numpy()

    # 텍스트 전처리
    reviews_cleaned = [preprocess_text(review) for review in tqdm(reviews)]

    splits = split_reviews(reviews_cleaned, labels, train_ratio, val_ratio)

    # BERT tokenization
    # uncased: 입력 단어를 소문자로 만들고 accent marks를 없앰
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    # LongformerTokenizer: Transformer 기반, 4096토큰까지 처리 가능
    # tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    # 인코딩
    '''
    padding: 배치에서 가장 긴 시퀀스를 기준으로 padding
    max_length
    truncation: max length보다 길 때 자르는 방법(true/default: 뒤를 자름)
    '''

    bert_arrays = {}

    for split_name in ['train', 'val', 'test']:
        input_ids, attention_mask = encode_split(
            splits[split_name]['reviews'],
            tokenizer_bert,
            batch_size=500,
            max_length=256
        )
        
        bert_arrays[split_name] = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': splits[split_name]['labels']
        }

    # RNN tokenization
    tokenizer_rnn = RNNTokenizer(vocab_size=vocab_size)
    tokenizer_rnn.build_vocab(splits['train']['reviews'])
    tokenizer_rnn.save_vocab("./data/tokenizer_rnn.json")

    rnn_arrays = {}

    for split_name in ['train', 'val', 'test']:
        input_ids = encode_split_rnn(
            splits[split_name]['reviews'],
            tokenizer_rnn,
            max_length=512
        )

        rnn_arrays[split_name] = {
            'input_ids' : input_ids,
            'attention_mask' : None,
            'labels' : splits[split_name]['labels']
        }

    save_array(bert_arrays, data_dir, data_type='BERT')
    save_array(rnn_arrays, data_dir, data_type="RNN")
    
    return bert_arrays, rnn_arrays


def save_array(arrays, data_dir='./data', data_type=''):
    os.makedirs(data_dir, exist_ok=True)
    
    print("Saving IMDB train data...")
    np.savez_compressed(
        os.path.join(data_dir, f'imdb_train_{data_type}.npz'),
        input_ids = arrays['train']['input_ids'],
        attention_mask = arrays['train']['attention_mask'],
        labels = arrays['train']['labels']
    )

    print("Saving IMDB validation data...")
    np.savez_compressed(
        os.path.join(data_dir, f'imdb_val_{data_type}.npz'),
        input_ids = arrays['val']['input_ids'],
        attention_mask = arrays['val']['attention_mask'],
        labels = arrays['val']['labels']
    )

    print("Saving IMDB test data...")
    np.savez_compressed(
        os.path.join(data_dir, f'imdb_test_{data_type}.npz'),
        input_ids = arrays['test']['input_ids'],
        attention_mask = arrays['test']['attention_mask'],
        labels = arrays['test']['labels']
    )


def check_dataset_distrib():
    df = pd.read_csv('./data/IMDB Dataset.csv')
        
    reviews = df['review'].tolist()

    # 텍스트 전처리
    reviews_cleaned = [preprocess_text(review) for review in tqdm(reviews)]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 토큰 길이 분포 확인
    token_lengths = []
    for review in reviews_cleaned:
        tokens = tokenizer.encode(review, add_special_tokens=True)
        token_lengths.append(len(tokens))

    import numpy as np
    print(f"평균: {np.mean(token_lengths)}")
    print(f"중간값: {np.median(token_lengths)}")
    print(f"95 percentile: {np.percentile(token_lengths, 95)}")
    print(f"최대: {np.max(token_lengths)}")
    print(f"512 초과: {sum(1 for l in token_lengths if l > 512)} / {len(token_lengths)}")

    '''
    평균: 253.28696
    중간값: 190.0
    95 percentile: 647.0
    최대: 2734  
    512 초과: 4594 / 50000

    BERT의 token 제한은 512개인데 전체 데이터의 약 9.2%가 이를 초과
    '''

if __name__ == "__main__":
    bert_arrays, rnn_arrays = preprocess_data('./data/IMDB Dataset.csv', data_dir)

    print("\n=== Summary ===")
    print("\n[BERT]")
    print(f"Train: {bert_arrays['train']['input_ids'].shape}")
    print(f"Val: {bert_arrays['val']['input_ids'].shape}")
    print(f"Test: {bert_arrays['test']['input_ids'].shape}")
    
    print("\n[RNN]")
    print(f"Train: {rnn_arrays['train']['input_ids'].shape}")
    print(f"Val: {rnn_arrays['val']['input_ids'].shape}")
    print(f"Test: {rnn_arrays['test']['input_ids'].shape}")

