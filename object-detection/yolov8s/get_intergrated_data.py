from imports import *
from coco_preprocessing import cooco_preprocessing
def get_integrated_data(ann_dir):
    #경로 설정
    base_path = r"C:\Users\KIMJW\Desktop\medicine\data"
    ann_dir = os.path.join(base_path,"raw" ,"train_annotations_1")
    json_paths = glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive = True)

    # 통합 딕셔너리
    # {이미지명 : {"boxes": [], "labels": [], "width": 0, "height": 0}} 구조
    integrated_data = defaultdict(lambda: {"boxes": [], "labels": [], "width": 0, "height": 0})

    for jp in json_paths:
        with open(jp, "r", encoding= "utf-8") as f:
            data = json.load(f)
        img_info = data["images"][0]
        f_name = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        # 이미지 기본 정보 저장
        integrated_data[f_name]["width"] = w
        integrated_data[f_name]["height"] = h

        # 어노테이션 정보 통합
        for ann in data["annotations"]:
            # 이미지 경계를 벗어나지 않도록 조정
            x, y, bw, bh = ann["bbox"]

            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            clean_box = [x, y, x + bw, y + bh] # 모델 학습용 XYXY포맷으로 변환

            if clean_box not in integrated_data[f_name]["boxes"]:
                integrated_data[f_name]["boxes"].append(clean_box)
                integrated_data[f_name]["labels"].append(ann["category_id"])

    return integrated_data

if __name__ == "__main__":
    BASE_PATH = r"C:\Users\KIMJW\Desktop\medicine\data"
    ANN_DIR = os.path.join(BASE_PATH, "raw","train_annotations_1")
    cooco_preprocessing()
    pill_data = get_integrated_data(ANN_DIR)

# 1. 박스가 아예 없는 이미지 (진짜 꽝)
    empty_images = [name for name, info in pill_data.items() if len(info['boxes']) == 0]

    # 2. 박스는 있지만, 크기가 너무 작아(예: 2픽셀 이하) 실질적으로 의미 없는 이미지
    small_box_images = []
    for name, info in pill_data.items():
        for box in info['boxes']:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width < 2 or height < 2: # 기준을 2픽셀로 설정
                small_box_images.append(name)
                break # 한 장이라도 있으면 추가

    print(f"❌ 박스 자체가 없는 이미지: {len(empty_images)}개")
    print(f"⚠️ 너무 작아서 안 보이는 박스가 포함된 이미지: {len(small_box_images)}개")
    print(f"integrated_data : {len(pill_data)}")
    if small_box_images:
        print(f"📋 작은 박스 파일 리스트(상위 5개): {small_box_images[:5]}")
