"""
YOLOv8s 추론 및 제출 CSV 생성
사용법: python model/yolov8s/yolo_inference.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from configs.load_paths import DATA_TEST_IMAGES, PROJECT_ROOT
import re


def find_best_model():
    """최신 YOLO 학습 결과에서 best.pt를 자동으로 찾기"""
    search_dirs = [
        PROJECT_ROOT / "outputs" / "yolo",
        PROJECT_ROOT / "pill_project",
        PROJECT_ROOT / "runs" / "detect",
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # 가장 최근 수정된 best.pt 찾기
        best_files = sorted(search_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if best_files:
            return best_files[0]
    return None


def yolo_inference(model_path=None, output_name=None):
    # 1. 학습된 모델 로드
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            print("모델 파일을 찾을 수 없습니다!")
            print("먼저 yolov8s_model.py로 학습을 실행하세요.")
            return

    print(f"모델 로딩: {model_path}")
    model = YOLO(model_path)
    model_to_catid = {}

    for k, v in model.names.items():
        numeric_id = re.sub(r'[^0-9]', '', v) 
        model_to_catid[int(k)] = int(numeric_id)


    # 2. 테스트 이미지 경로 및 파일 목록 확보 (공통 경로 사용)
    test_img_dir = DATA_TEST_IMAGES
    test_images = sorted([f.name for f in test_img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

    submission_data = []
    ann_id = 1

    print("🚀 추론 시작...")
    for img_name in tqdm(test_images):
        img_path = test_img_dir / img_name
        
        # [수정] image_id: 이미지 파일명에서 숫자만 추출하여 정수로 변환
        # 루프 안에서 현재 처리 중인 img_name을 사용해야 합니다.
        image_id = int(re.sub(r'[^0-9]', '', img_name))
        
        results = model.predict(img_path, conf=0.03,
                                 imgsz = 1024,
                                 verbose=False)
        
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, score, cls_id = box
                
                # [수정] category_id: YOLO 인덱스(0~55)에 1을 더해 대회 양식(1~56)에 맞춤
                model_label = int(cls_id)
                original_category_id = model_to_catid.get(model_label, model_label)
                # XYXY -> XYWH 변환
                w = x2 - x1
                h = y2 - y1
                
                # 유효하지 않은 박스 제외
                if w <= 0 or h <= 0:
                    continue

                submission_data.append({
                    "annotation_id": ann_id,
                    "image_id": image_id,
                    "category_id": original_category_id,  
                    "bbox_x": round(float(x1), 2),
                    "bbox_y": round(float(y1), 2),
                    "bbox_w": round(float(w), 2),
                    "bbox_h": round(float(h), 2),
                    "score": round(float(score), 4)
                })
                ann_id += 1 
                
    # 3. CSV 저장 (모델 이름으로 자동 생성)
    df = pd.DataFrame(submission_data)
    output_dir = PROJECT_ROOT / "submit"
    output_dir.mkdir(exist_ok=True)
    if output_name is None:
        # 모델 경로에서 이름 추출 (예: outputs/yolo/yolo1/weights/best.pt → yolo1)
        output_name = Path(model_path).parents[1].name + ".csv"
    output_path = output_dir / output_name
    df.to_csv(output_path, index=False)
    print(f"✅ 제출 파일 생성 완료: {output_path}")

if __name__ == "__main__":
    yolo_inference()