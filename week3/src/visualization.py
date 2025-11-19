import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def plot_training_history(history, dataset_name, save_path=None):
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Loss
    plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{dataset_name} Training and Validation Loss', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def visualize_unk_ratio(reviews, tokenizer_rnn, max_length=512, save_path=None):
    unk_idx = tokenizer_rnn.word2idx["<unk>"]
    pad_idx = tokenizer_rnn.word2idx["<pad>"]
    sos_idx = tokenizer_rnn.word2idx.get("<sos>", -1) 
    eos_idx = tokenizer_rnn.word2idx.get("<eos>", -1)

    special_tokens = {pad_idx, sos_idx, eos_idx}

    total_tokens = 0
    unk_tokens = 0
    unk_ratios_per_review = []

    for review in tqdm(reviews, desc="Analysing UNK ratio"):
        tokens = tokenizer_rnn.encode(review, max_length)
        
        actual_tokens = [t for t in tokens if t not in special_tokens]
        
        total_tokens += len(actual_tokens)
        review_unk_count = actual_tokens.count(unk_idx)
        unk_tokens += review_unk_count

        if len(actual_tokens) > 0:
            unk_ratios_per_review.append(review_unk_count / len(actual_tokens) * 100)
        else:
            unk_ratios_per_review.append(0)

    overall_unk_ratio = (unk_tokens / total_tokens * 100 ) if total_tokens > 0 else 0

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 전체 UNK 비율
    axes[0].bar(['Known Tokens', 'UNK Tokens'], 
                [100 - overall_unk_ratio, overall_unk_ratio],
                color=['#2ecc71', '#e74c3c'])
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title(f'Overall Token Distribution\n(UNK: {overall_unk_ratio:.2f}%)')
    axes[0].set_ylim([0, 100])
    
    for i, v in enumerate([100 - overall_unk_ratio, overall_unk_ratio]):
        axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
    
    # 2. 리뷰별 UNK 비율
    axes[1].hist(unk_ratios_per_review, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('UNK Ratio per Review (%)')
    axes[1].set_ylabel('Number of Reviews')
    axes[1].set_title('Distribution of UNK Ratio Across Reviews')
    axes[1].axvline(np.mean(unk_ratios_per_review), color='red', 
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(unk_ratios_per_review):.2f}%')
    axes[1].legend()
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()
    
    # 통계 출력
    print("\n=== UNK Token Analysis ===")
    print(f"Total tokens: {total_tokens:,}")
    print(f"UNK tokens: {unk_tokens:,}")
    print(f"Overall UNK ratio: {overall_unk_ratio:.2f}%")
    print(f"\nPer-review UNK ratio statistics:")
    print(f"  Mean: {np.mean(unk_ratios_per_review):.2f}%")
    print(f"  Median: {np.median(unk_ratios_per_review):.2f}%")
    print(f"  Std: {np.std(unk_ratios_per_review):.2f}%")
    print(f"  Min: {np.min(unk_ratios_per_review):.2f}%")
    print(f"  Max: {np.max(unk_ratios_per_review):.2f}%")
    
    return overall_unk_ratio, unk_ratios_per_review