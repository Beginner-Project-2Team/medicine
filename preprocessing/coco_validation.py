"""
COCO 포맷 데이터 검증
YOLO, Faster R-CNN 등 모든 모델에서 공통으로 사용
사용법: python preprocessing/coco_validation.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from configs.load_paths import DATA_TRAIN_ANNOTATIONS, DATA_TRAIN_IMAGES, DATA_TEST_IMAGES
from preprocessing.coco_data_integration import get_integrated_coco_data

def coco_validation():
    """COCO 포맷 데이터의 품질을 검증합니다."""
    ann_dir = DATA_TRAIN_ANNOTATIONS
    img_dir = DATA_TRAIN_IMAGES
    test_dir = DATA_TEST_IMAGES

    json_paths = list(ann_dir.rglob("*.json"))

    issues = {
        "missing_images": [],      # 이미지 파일이 없는 경우
        "invalid_bboxes": [],       # 잘못된 바운딩 박스
        "format_errors": [],        # [x,y,w,h] 형식 오류
        "category_mismatch": [],    # 카테고리 불일치
        "out_of_bounds": []         # 바운딩 박스가 이미지 경계를 벗어나는 경우
    }

    valid_categories = set()

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
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

    print("==== COCO 데이터 검증 결과 ====")
    print(f"📂 JSON 파일 개수: {len(json_paths)}개")
    print(f"🖼️  Test 이미지 개수: {len(list(test_dir.glob('*')))}개")
    print(f"\n품질 이슈:")
    print(f"  ❌ 이미지 없는 정답지: {len(issues['missing_images'])}개")
    print(f"  ⚠️  불량 박스(크기 0 또는 음수): {len(issues['invalid_bboxes'])}개")
    print(f"  ⚠️  형식 오류 박스: {len(issues['format_errors'])}개")
    print(f"  ⚠️  카테고리 불일치: {len(issues['category_mismatch'])}개")
    print(f"  ⚠️  경계 벗어난 박스: {len(issues['out_of_bounds'])}개")

    if sum(len(v) for v in issues.values()) == 0:
        print("\n✅ 모든 검증 통과!")
    else:
        print(f"\n⚠️  총 {sum(len(v) for v in issues.values())}개 이슈 발견")

    return issues


if __name__ == "__main__":
    results = coco_validation()
