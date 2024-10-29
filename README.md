- 수정 불가한 file : detect.py, east_dataset.py, loss.py, model.py

1. train.py
    - Main 실행 파일입니다. WanDB 접속 등 다양한 custom option을 제공하며, 다른 코드의 내용을 적절하게 불러와 사용합니다.

2. dataset.py
    - dataset에 모든 EDA 기법을 적용시키는 script file 입니다.
    - Matrix Transform (Rotation), Denosing, Super Resolution 등등의 기법을 수행합니다.

3. tools
    - merge_ufo.py : ufo 폴더 안에 위치한 json file 들을 병합합니다. 병합할 내용은 'chinese_train.json', 'japanese_train.json', 'thai_train.json', 'vietnamese_train.json' 입니다. 병합된 결과는 'merged_ufo.json' 파일로 저장됩니다.
    - split_ufo.py : 'merged_ufo.json' 파일을 train과 validation set으로 나눕니다. 나눠진 결과는 'train1.json'과 'valid1.json' 파일로 저장됩니다.
    - 해당 코드를 정상적으로 실행한 뒤에, 'data' 폴더 이름을 'dataset'으로 변경해주세요. 정상적으로 실행된 폴더의 트리 구조는 아래와 같습니다.

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

4. Error
```
Traceback (most recent call last):
  File "/data/ephemeral/home/.GitHub/level2-cv-datacentric-cv-21/train.py", line 254, in <module>
    main(args)
  File "/data/ephemeral/home/.GitHub/level2-cv-datacentric-cv-21/train.py", line 248, in main
    do_training(args, **args.__dict__)
  File "/data/ephemeral/home/.GitHub/level2-cv-datacentric-cv-21/train.py", line 146, in do_training
    for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1345, in _next_data
    return self._process_data(data)
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1371, in _process_data
    data.reraise()
  File "/opt/conda/lib/python3.10/site-packages/torch/_utils.py", line 694, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/data/ephemeral/home/.GitHub/level2-cv-datacentric-cv-21/east_dataset.py", line 136, in __getitem__
    image, word_bboxes, roi_mask = self.dataset[idx]
  File "/data/ephemeral/home/.GitHub/level2-cv-datacentric-cv-21/dataset.py", line 540, in __getitem__
    word_tags = word_info['tags']
KeyError: 'tags'
```