from imports import *
from configs.load_paths import DATA_TRAIN_ANNOTATIONS, DATA_TRAIN_IMAGES, DATA_TEST_IMAGES

def yolo_cooco_preprocessing():
    ann_dir = DATA_TRAIN_ANNOTATIONS
    img_dir = DATA_TRAIN_IMAGES
    test_dir = DATA_TEST_IMAGES

    json_paths = list(ann_dir.rglob("*.json"))

    issues = {
        
        "missing_images" : [], # 이미지 파일이 없는 경우
        "invalid_bboxes" : [], # 잘못된 바운딩 박스
        "format_errors" : [], # [x,y,w,h] 형식 오류
        "category_mismatch" : [], # 카테고리 불일치
        "out_of_bounds" : [] # 바운딩 박스가 이미지 경계를 벗어나는 경우
    }

    valid_categories = set()

    for jp in json_paths:
        with open(jp, "r", encoding= "utf-8") as f:
            data = json.load(f)

        img_info = data["images"][0]
        f_name = img_info["file_name"]
        img_path = img_dir / f_name

        # 이미지 존재 여부 확인
        if not img_path.exists():
            issues["missing_images"].append(f_name)
            continue

        width, height = img_info["width"], img_info["height"]

        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            category_id = ann["category_id"]

            # 카테고리 유효성 검사
            valid_categories.add(category_id)

            # 바운딩 박스 형식 검사
            if w <= 0 or h <= 0:
                issues["format_errors"].append((f_name, ann))
                continue

            # 바운딩 박스 경계 검사
            if x < 0 or y < 0 or x + w > width or y + h > height:
                issues["out_of_bounds"].append((f_name, ann))

    print("----coco 데이터 전처리 결과----")
    print(f"사진 없는 정답지: {len(issues['missing_images'])}개")
    print(f"불량 박스(크기 0 또는 음수): {len(issues['invalid_bboxes'])}개")
    print(f"형식 오류 박스: {len(issues['format_errors'])}개")
    print(f"카테고리 불일치 박스: {len(issues['category_mismatch'])}개")
    print(f"경계 벗어난 박스: {len(issues['out_of_bounds'])}개")
    print(f"json_files 개수:{len(json_paths)}")
    print(f"test_img 개수 : {len(os.listdir(test_dir))}")

    return issues

if __name__=="__main__":
    results = yolo_cooco_preprocessing()

