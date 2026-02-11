import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (import 경로 해결)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from configs.load_paths import DATA_TRAIN_ANNOTATIONS, DATA_PROCESSED

def get_integrated_coco_data(ann_dir=None, use_cache=True):
    if ann_dir is None:
        ann_dir = DATA_TRAIN_ANNOTATIONS

    # Path 객체로 변환
    ann_dir = Path(ann_dir) if not isinstance(ann_dir, Path) else ann_dir

    # 캐시 파일 경로
    cache_file = DATA_PROCESSED / "integrated_coco_data.json"

    # 캐시 사용 & 캐시 파일 존재 시 로드
    if use_cache and cache_file.exists():
        print(f"📂 캐시된 데이터 로딩 중... ({cache_file})")
        with open(cache_file, "r", encoding="utf-8") as f:
            integrated_data = json.load(f)
        print(f"✅ 캐시 로드 완료: {len(integrated_data)}개 이미지")
        return integrated_data

    # 캐시 없음 → 새로 병합
    json_paths = list(ann_dir.rglob("*.json"))
    integrated_data = {}

    print(f"📦 데이터를 통합 중... ({len(json_paths)}개 JSON 파일)")

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_info = data["images"][0]
        f_name = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        # 이미지별 딕셔너리 초기화
        if f_name not in integrated_data:
            integrated_data[f_name] = {"boxes": [], "labels": [], "width": w, "height": h}

        # 어노테이션 정보 통합
        for ann in data["annotations"]:
            # 이미지 경계를 벗어나지 않도록 조정
            x, y, bw, bh = ann["bbox"]

            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            # 모델 학습용 XYXY 포맷으로 변환
            clean_box = [x, y, x + bw, y + bh]

            # 중복 박스 방지
            if clean_box not in integrated_data[f_name]["boxes"]:
                integrated_data[f_name]["boxes"].append(clean_box)
                integrated_data[f_name]["labels"].append(ann["category_id"])

    print(f"✅ 통합 완료: {len(integrated_data)}개 이미지")

    # 캐시 파일로 저장
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(integrated_data, f, indent=2, ensure_ascii=False)
    print(f"💾 캐시 저장 완료: {cache_file}")

    return integrated_data


if __name__ == "__main__":
    # 테스트용
    pill_data = get_integrated_coco_data()

    # 1. 박스가 아예 없는 이미지
    empty_images = [name for name, info in pill_data.items() if len(info['boxes']) == 0]

    # 2. 박스는 있지만, 크기가 너무 작은 이미지
    small_box_images = []
    for name, info in pill_data.items():
        for box in info['boxes']:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width < 2 or height < 2:
                small_box_images.append(name)
                break

    print(f"\n📊 데이터 품질 체크:")
    print(f"❌ 박스 자체가 없는 이미지: {len(empty_images)}개")
    print(f"⚠️  너무 작은 박스가 포함된 이미지: {len(small_box_images)}개")
    if small_box_images:
        print(f"📋 작은 박스 파일 리스트(상위 5개): {small_box_images[:5]}")
