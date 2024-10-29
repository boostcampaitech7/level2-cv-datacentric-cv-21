- 수정 불가한 file : detect.py, east_dataset.py, loss.py, model.py

1. train.py
    - Main 실행 파일입니다. WanDB 접속 등 다양한 custom option을 제공하며, 다른 코드의 내용을 적절하게 불러와 사용합니다.
    - 이 파일을 실행하기 위해서는 '../data' 안에 train dataset이 위치해야 합니다.
<br/>
2. dataset.py
    - dataset에 모든 EDA 기법을 적용시키는 script file 입니다.
    - Matrix Transform (Rotation), Denosing, Super Resolution 등등의 기법을 수행합니다.
<br/>
3. 'train.py'를 실행하기 위해서는 아래의 tree 구조가 필요합니다.


```
📦.GitHub
 ┣ 📂.git
 ┣ 📂.github
 ┣ 📂baseline / ~
 ┣ 📂pths
 ┃ ┗ 📜vgg16_bn-6c64b313.pth
 ┣ 📂tools / ~ : 아직 사용할 필요 없는 내용들입니다.
 ┣ 📂utils / ~
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜dataset.py : 전체적인 전처리 과정을 수행하는 파일입니다.
 ┣ 📜deteval.py
 ┣ 📜inference.py : checkpoint를 불러옵니다.
 ┣ 📜metrics.py
 ┣ 📜requirements.txt
 ┣ 📜train.py : main 실행 파일입니다.
 ┗ 📜utils.py
```
