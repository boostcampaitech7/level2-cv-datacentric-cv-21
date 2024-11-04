import os
import re
import glob
from pathlib import Path


def increment_path(path):  # 중복된 이름 있으면 이름 뒤에 숫자 붙여서 구분함
    path = Path(path)
    if not path.exists():
        os.makedirs(path)
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = f"{path}{n}"
        os.makedirs(path)
        return path