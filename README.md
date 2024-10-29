- 수정 불가한 file : detect.py, east_dataset.py, loss.py, model.py

1. train.py
    - Main 실행 파일입니다. WanDB 접속 등 다양한 custom option을 제공하며, 다른 코드의 내용을 적절하게 불러와 사용합니다.
    - 이 파일을 실행하기 위해서는 '../data' 안에 train dataset이 위치해야 합니다.

2. dataset.py
    - dataset에 모든 EDA 기법을 적용시키는 script file 입니다.
    - Matrix Transform (Rotation), Denosing, Super Resolution 등등의 기법을 수행합니다.


```
📦.GitHub
 ┣ 📂.git
 ┣ 📂.github
 ┣ 📂baseline
 ┃ ┣ 📜detect.py
 ┃ ┣ 📜east_dataset.py
 ┃ ┣ 📜loss.py
 ┃ ┗ 📜model.py
 ┣ 📂pths
 ┃ ┗ 📜vgg16_bn-6c64b313.pth
 ┣ 📂tools
 ┃ ┣ 📜merge_ufo.py
 ┃ ┗ 📜train_val_split.py
 ┣ 📂utils
 ┃ ┣ 📜coco2ufo.ipynb
 ┃ ┣ 📜ensemble.ipynb
 ┃ ┣ 📜lift_up_bounding_boxes.ipynb
 ┃ ┣ 📜manual_k_fold.ipynb
 ┃ ┣ 📜ufo2coco.ipynb
 ┃ ┣ 📜visualization.ipynb
 ┃ ┗ 📜weighted_boxes_fusion.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜dataset.py
 ┣ 📜deteval.py
 ┣ 📜inference.py
 ┣ 📜metrics.py
 ┣ 📜requirements.txt
 ┣ 📜train.py
 ┗ 📜utils.py
```

4. Error
```

```
