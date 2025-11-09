import numpy as np

data = np.load('./data/imdb_train_RNN.npz')
print("Keys in RNN file:", data.files)
print("input_ids shape:", data['input_ids'].shape)

# attention_mask가 있나요?
if 'attention_mask' in data.files:
    print("attention_mask exists:", data['attention_mask'])