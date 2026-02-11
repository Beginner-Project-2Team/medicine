"""
YOLOv8s 학습 스크립트
사용법: python model/yolov8s/yolov8s_model.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from configs.load_paths import DATASET_YOLO

def train_model(epochs=2, imgsz=640, batch=4, name="yolo1"):
    # 1. 모델 로드
    model = YOLO('yolov8s.pt')

    # 2. 데이터셋 경로 설정 (공통 경로 사용)
    data_yaml_path = DATASET_YOLO / "data.yaml"

    # 3. 모델 학습 (Training)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=str(PROJECT_ROOT / "outputs" / "yolo"),
        name=name,
    )

    # 4. 성능 검증
    metrics = model.val()
    print(f"Mean Average Precision (mAP): {metrics.box.map}")

# --- 이 부분이 가장 중요합니다! ---
if __name__ == '__main__':
    # 윈도우 멀티프로세싱 에러를 방지하기 위해
    # 반드시 이 안에서 실행 코드를 호출해야 합니다.
    train_model(name="yolo1")  # ← 개별 실행 시 여기서 이름 변경