import numpy as np
import os
from src.vocabulary import Tokenizer

def test_saved_data():
    """저장된 데이터 간단 검증 + 디코딩"""
    print("="*50)
    print("Testing Saved Data")
    print("="*50)
    
    phases = ['train', 'validation', 'test']
    
    for phase in phases:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data")
        data_dir = os.path.abspath(data_dir)

        file_path = os.path.join(data_dir, f'cnn_{phase}.npz')
        
        # 파일 존재 확인
        assert os.path.exists(file_path), f"❌ {phase} file not found!"
        
        # 데이터 로드
        data = np.load(file_path)
        
        # 검증
        assert 'articles' in data and 'highlights' in data, f"❌ Missing keys in {phase}"
        assert data['articles'].shape[0] == data['highlights'].shape[0], f"❌ Shape mismatch in {phase}"
        assert data['articles'].dtype == np.int64, f"❌ Wrong dtype in {phase}"
        
        # 출력
        print(f"\n✅ {phase.upper()}: {data['articles'].shape[0]} samples")
        print(f"   Articles: {data['articles'].shape}, range [{data['articles'].min()}, {data['articles'].max()}]")
        print(f"   Highlights: {data['highlights'].shape}, range [{data['highlights'].min()}, {data['highlights'].max()}]")
    
    # 디코딩 테스트 (train 데이터 처음 5개)
    print("\n" + "="*50)
    print("Decoding First 5 Samples (Train)")
    print("="*50)
    
    tokenizer = Tokenizer(vocab_size=20000)
    tokenizer.load_vocab(os.path.join(data_dir, 'tokenizer.json'))
    
    train_data = np.load(os.path.join(data_dir, 'cnn_test.npz'))
    
    for i in range(min(5, len(train_data['articles']))):
        article_text = tokenizer.decode(train_data['articles'][i].tolist())[:100]
        highlight_text = tokenizer.decode(train_data['highlights'][i].tolist())[:100]
        
        print(f"\n[Sample {i+1}]")
        print(f"Article:   {article_text}...")
        print(f"Highlight: {highlight_text}...")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_saved_data()

