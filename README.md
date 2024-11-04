1. baseline :
    - 대회의 규정상 수정이 불가한 파일들을 모아둔 폴더입니다.
    - detect.py, east_dataset.py, loss.py, model.py 가 포함되어 있습니다.
2. train.py
    - Main 실행 파일입니다. WanDB 접속 등 다양한 custom option을 제공하며, 다른 코드의 내용을 적절하게 불러와 사용합니다.
    - 이 파일을 실행하기 위해서는 '../data' 경로에 train dataset이 위치해야 합니다.
3. dataset.py : dataset에 다양한 EDA 기법을 적용시키는 script file 입니다.
4. archive : 이전 기수(6기)에서 1위를 기록했던 팀의 solution입니다.
5. 'train.py'를 실행하기 위해서는 아래의 조건을 만족해야 합니다.
    - Baseline code를 다운로드 받는다면 얻을 수 있는, "vgg16_bn-6c64b313.pth" 파일이 필요합니다.
    -  "/data/ephemeral/home/" 경로에 "data" 폴더가 필요합니다.



```
📦github
 ┣ 📂.git
 ┣ 📂.github
 ┣ 📂archives : 이전 기수 분들의 자료입니다.
 ┣ 📂baseline : 수정이 금지된 파일들의 저장소입니다.
 ┣ 📂pths
 ┃ ┗ 📜vgg16_bn-6c64b313.pth
 ┣ 📜.gitignore
 ┣ 📜dataset.py : 전체적인 전처리 과정을 수행하는 파일입니다.
 ┣ 📜deteval.py
 ┣ 📜inference.py : checkpoint를 불러옵니다.
 ┣ 📜optimizer.py
 ┣ 📜requirements.txt
 ┣ 📜train.py : main 실행 파일입니다.
 ┗ 📜utils.py
```
