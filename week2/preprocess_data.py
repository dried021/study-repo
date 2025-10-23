import pandas as pd
from transformers import BertTokenizer, LongformerTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os

from config import data_dir, train_ratio, val_ratio
import re

# 개별 텍스트 전처리 함수
def preprocess_text(text):
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


def preprocess_data(file_path):
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

    # uncased: 입력 단어를 소문자로 만들고 accent marks를 없앰
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 최대 512 토큰까지 처리 가능

    # LongformerTokenizer: Transformer 기반, 4096토큰까지 처리 가능
    # tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    # 인코딩
    '''
    padding: 배치에서 가장 긴 시퀀스를 기준으로 padding
    max_length
    truncation: max length보다 길 때 자르는 방법(true/default: 뒤를 자름)
        입력이 쌍일 경우 
    '''

    batch_size = 500 # 한 번에 500개씩 처리
    encoded_list = []

    for i in tqdm(range(0, len(reviews_cleaned), batch_size)):  
        batch = reviews_cleaned[i: i+batch_size]
        encoded_batch = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=512, # 최대 토큰 길이는 2734이지만 BERT 한계
            return_tensors='np'
        )
        encoded_list.append(encoded_batch)

    input_ids = np.concatenate([e['input_ids'] for e in encoded_list])
    attention_mask = np.concatenate([e['attention_mask'] for e in encoded_list])

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention Mask shape: {attention_mask.shape}")
    print(f"Number of labels: {len(labels)}")

    return input_ids, attention_mask, labels

def split_data(input_ids, attention_mask, labels, train_ratio = 0.8, val_ratio = 0.2):
    '''
    데이터 분할
    '''
    # train / test 분할
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=(1 - train_ratio),
        random_state=42,
        stratify=labels
    )

    X_train_ids, X_test_ids = input_ids[train_idx], input_ids[test_idx]
    X_train_mask, X_test_mask = attention_mask[train_idx], attention_mask[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # train/val 분할 
    train_idx2, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    X_val_ids, X_val_mask, y_val = (
        X_train_ids[val_idx],
        X_train_mask[val_idx],
        y_train[val_idx],
    )
    X_train_ids, X_train_mask, y_train = (
        X_train_ids[train_idx2],
        X_train_mask[train_idx2],
        y_train[train_idx2],
    )

    arrays = {
        'train': {
            'input_ids': X_train_ids,
            'attention_mask': X_train_mask,
            'labels': y_train
        },
        'val': {
            'input_ids': X_val_ids,
            'attention_mask': X_val_mask,
            'labels': y_val
        },
        'test': {
            'input_ids': X_test_ids,
            'attention_mask': X_test_mask,
            'labels': y_test
        }
    }
    return arrays

def save_array(arrays, data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)
    
    print("Saving IMDB train data...")
    np.savez_compressed(
        os.path.join(data_dir, 'imdb_train.npz'),
        input_ids = arrays['train']['input_ids'],
        attention_mask = arrays['train']['attention_mask'],
        labels = arrays['train']['labels']
    )

    print("Saving IMDB validation data...")
    np.savez_compressed(
        os.path.join(data_dir, 'imdb_val.npz'),
        input_ids = arrays['val']['input_ids'],
        attention_mask = arrays['val']['attention_mask'],
        labels = arrays['val']['labels']
    )

    print("Saving IMDB test data...")
    np.savez_compressed(
        os.path.join(data_dir, 'imdb_test.npz'),
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

    BERT의 token 제한은 512개인데 전체 데이터의 약 9.2%가 이를 초과 -> BERT 사용이 적절하지 않음
    '''

if __name__ == "__main__":
    input_ids, attention_mask, labels = preprocess_data('./data/IMDB Dataset.csv')
    arrays = split_data(input_ids, attention_mask, labels, train_ratio, val_ratio)
    save_array(arrays, data_dir)

    print(f"\nTrain: {arrays['train']['input_ids'].shape}")
    print(f"Val: {arrays['val']['input_ids'].shape}")
    print(f"Test: {arrays['test']['input_ids'].shape}")