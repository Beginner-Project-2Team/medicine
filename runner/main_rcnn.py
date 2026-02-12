"""
Faster R-CNN 파이프라인
사용법:
  python runner/main_rcnn.py              → 학습만
  python runner/main_rcnn.py --inference  → 추론 + CSV 생성
  python runner/main_rcnn.py y --all        → 학습 + 추론
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.faster_rcnn.rcnn_train import run_training
from model.faster_rcnn.rcnn_inference import rcnn_inference

# ============================================================
# 하이퍼파라미터 (여기서 수정하면 각 모듈로 전달됨)
# ============================================================
MODEL_NAME = "rcnn1"       # 모델 이름 (실험마다 바꾸기)
BATCH_SIZE = 4
NUM_EPOCHS = 2
LR = 0.001
USE_AUG = False             # 증강 데이터 사용 여부

# 추론
CONF_THRESHOLD = 0.25      # confidence threshold

# 실행 모드
DO_TRAIN = True            # 학습
DO_INFERENCE = False       # 추론 + csv
# ============================================================



def main():
    if DO_TRAIN:
        print("=" * 50)
        print("  [1/2] Faster R-CNN 학습")
        print("=" * 50)
        run_training(
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            name=MODEL_NAME,
            use_aug=USE_AUG,
        )

    if DO_INFERENCE:
        print("=" * 50)
        print("  [2/2] Faster R-CNN 추론 → CSV 생성")
        print("=" * 50)
        rcnn_inference(conf_threshold=CONF_THRESHOLD)


if __name__ == "__main__":
    main()
