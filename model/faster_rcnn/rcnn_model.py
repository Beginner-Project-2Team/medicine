"""
Faster R-CNN 모델 정의
ResNet50 + FPN (Feature Pyramid Network) 백본 사용
사용법: python model/faster_rcnn/rcnn_model.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes: int):
   
    # [1단계] COCO로 사전학습된 Faster R-CNN 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="COCO_V1"  # 사전학습 가중치 (전이학습)
    )

    # [2단계] 마지막 분류기를 우리 데이터에 맞게 교체
    # 현재 분류기의 입력 크기 확인
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 새로운 분류기로 교체
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


def get_num_classes(catid_to_model: dict):
    # 모델 label의 최대값 찾기 (보통 56)
    # +1 하면 배경(0) 포함 클래스 개수 (57)
    max_label = max(catid_to_model.values())
    return max_label + 1


if __name__ == "__main__":
    # 테스트: 모델 생성해보기
    test_model = create_model(num_classes=57)
    print(f"✅ 모델 생성 완료!")
    print(f"📊 파라미터 개수: {sum(p.numel() for p in test_model.parameters()):,}개")

    # 간단한 forward test
    dummy_input = [torch.randn(3, 640, 640)]
    test_model.eval()
    with torch.no_grad():
        output = test_model(dummy_input)
    print(f"✅ Forward pass 성공!")
