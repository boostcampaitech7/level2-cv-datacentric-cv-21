import os
import json
from glob import glob

def merge_json_files(folder_path):
    merged_data = {"images": {}}

    # 폴더 내 모든 json 파일을 읽기
    json_files = glob(os.path.join(folder_path, "*.json"))

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            # 파일 안의 "images" 데이터를 병합
            if "images" in data:
                for image_name, content in data["images"].items():
                    if image_name not in merged_data["images"]:
                        merged_data["images"][image_name] = content
                    else:
                        # 이미지가 이미 있을 경우, 해당 이미지의 데이터 병합
                        merged_data["images"][image_name]["paragraphs"].update(content.get("paragraphs", {}))
                        merged_data["images"][image_name]["words"].update(content.get("words", {}))

    # 결과를 병합된 json 파일로 저장
    with open(os.path.join(folder_path, "merged_data.json"), 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)

# 실행 예시
merge_json_files("/data/ephemeral/home/.GitHub/dataset/ufo/train")
