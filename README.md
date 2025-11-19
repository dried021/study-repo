# study-repo

## 패키지 설치
pip install -r requirements.txt

## 데이터 준비
python download_data.py 
또는
python download_data.py --save_dir [경로] --dataset [mnist/cifar10/both]


## 학습 실행
python main_train.py

## 테스트 실행
python main_test.py