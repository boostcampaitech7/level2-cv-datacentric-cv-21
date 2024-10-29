- 수정 불가한 file : detect.py, east_dataset.py, loss.py, model.py
- Model Training method
  - 4개 언어 dataset을 병합한 뒤 학습하는 방식
  - 4개 언어 dataset에 대한 각각의 model을 생성한 뒤, inference를 통해서 결합해 최종 model을 생성하는 방식

1. train.py
    - Main 실행 파일입니다. WanDB 접속 등 다양한 custom option을 제공하며, 다른 코드의 내용을 적절하게 불러와 사용합니다.

2. dataset.py
    - dataset에 모든 EDA 기법을 적용시키는 script file 입니다.
    - Matrix Transform (Rotation), Denosing, Super Resolution 등등의 기법을 수행합니다.

3. tools (Optional)
    - merge_ufo.py : ufo 폴더 안에 위치한 json file 들을 병합합니다. 병합할 내용은 'chinese_train.json', 'japanese_train.json', 'thai_train.json', 'vietnamese_train.json' 입니다. 병합된 결과는 'merged_ufo.json' 파일로 저장됩니다.
    - split_ufo.py : 'merged_ufo.json' 파일을 train과 validation set으로 나눕니다. 나눠진 결과는 'train1.json'과 'valid1.json' 파일로 저장됩니다.
    - 해당 코드를 정상적으로 실행한 뒤에, 'data' 폴더 이름을 'dataset'으로 변경해주세요. 정상적으로 실행된 폴더의 트리 구조는 아래와 같습니다.

```
    📦dataset
    ┗ 📂ufo
    ┃ ┣ 📂test
    ┃ ┣ 📂train
    ┃ ┃ ┣ 📜chinese_train.json
    ┃ ┃ ┣ 📜japanese_train.json
    ┃ ┃ ┣ 📜thai_train.json
    ┃ ┃ ┣ 📜train.json
    ┃ ┃ ┗ 📜vietnamese_train.json
    ┃ ┣ 📜train1.json : 실제 학습용 dataset 입니다.
    ┃ ┗ 📜valid1.json : 실제 검증용 dataset 입니다.
```

4. Error
```

```

5. To-Do : 4개 언어 별로 학습된 model의 inference를 결합하는 방식으로 최종 model을 생성하는 방식을 구현 필요.