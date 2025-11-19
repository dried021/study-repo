import numpy as np
from transformers import LongformerTokenizer

# 토크나이저 로드
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

train_data = np.load('../data/imdb_train.npz')
input_ids = train_data['input_ids']
labels = train_data['labels']

print("\n=== first 5 reviews ===")
# for i in range(100):
#     review_text = tokenizer.decode(input_ids[i])
#     print(f"\n[Review {i}]")
#     print(review_text[:100] + "...")  # 앞 200자만
#     print(labels[i])

# numpy array용 확인 방법
print(f"Unique labels: {np.unique(labels)}")
print(f"Label counts:")
unique, counts = np.unique(labels, return_counts=True)
for val, cnt in zip(unique, counts):
    print(f"  {val}: {cnt}")

print(f"Min: {np.min(labels)}, Max: {np.max(labels)}")
print(f"Total samples: {len(labels)}")

# 0, 1 외의 값이 있는지 확인
invalid_mask = (labels != 0) & (labels != 1)
invalid_labels = labels[invalid_mask]
if len(invalid_labels) > 0:
    print(f"Invalid labels found: {invalid_labels}")
    print(f"Number of invalid labels: {len(invalid_labels)}")
else:
    print("All labels are valid (0 or 1)")