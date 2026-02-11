"""
YOLOv8s 파이프라인
사용법:
  python main_yolo.py              → 변환 + 학습
  python main_yolo.py --inference  → 추론 + CSV 생성
  python main_yolo.py --all        → 변환 + 학습 + 추론
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.yolo_converter import PillYOLOConverter
from model.yolov8s.yolov8s_model import train_model
from model.yolov8s.yolo_inference import yolo_inference

# 모델 이름 (실험마다 바꾸기) 
MODEL_NAME = "yolo1"
# 

# 하이퍼파라미터 (여기서 수정)
EPOCHS = 2
IMGSZ = 640
BATCH = 4

# 실행 모드 (True/False로 선택)
DO_CONVERT = True       # COCO → YOLO 포맷 변환
DO_TRAIN = True         # 모델 학습
DO_INFERENCE = False    # 추론 + csv



def main():
    if DO_CONVERT:
        print("=" * 50)
        print("  [1/3] COCO → YOLO 데이터 변환")
        print("=" * 50)
        converter = PillYOLOConverter()
        converter.split_check()

    if DO_TRAIN:
        print("=" * 50)
        print("  [2/3] YOLOv8s 학습")
        print("=" * 50)
        train_model(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, name=MODEL_NAME)

    if DO_INFERENCE:
        print("=" * 50)
        print("  [3/3] YOLOv8s 추론 → CSV 생성")
        print("=" * 50)
        yolo_inference()


if __name__ == "__main__":
    main()
