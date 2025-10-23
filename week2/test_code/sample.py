import numpy as np
from transformers import LongformerTokenizer

# 토크나이저 로드
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

train_data = np.load('../data/imdb_train.npz')
input_ids = train_data['input_ids']
labels = train_data['labels']

print("\n=== first 5 reviews ===")
for i in range(5):
    review_text = tokenizer.decode(input_ids[i])
    print(f"\n[Review {i}]")
    print(review_text[:200] + "...")  # 앞 200자만
    print(labels[i])