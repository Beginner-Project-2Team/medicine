from imports import *
from configs.load_paths import DATASET_YOLO

def train_model():
    # 1. 모델 로드
    model = YOLO('yolov8s.pt')

    # 2. 데이터셋 경로 설정 (공통 경로 사용)
    data_yaml_path = DATASET_YOLO / "data.yaml"

    # 3. 모델 학습 (Training)
    results = model.train(
        data=data_yaml_path,   
        epochs=50,             
        imgsz=640,             
        batch=16,              
        device=0,              # CUDA 사용을 위해 0으로 설정
        project='pill_project', 
        name='yolov8_pill_detect'
    )

    # 4. 성능 검증
    metrics = model.val()
    print(f"Mean Average Precision (mAP): {metrics.box.map}")

# --- 이 부분이 가장 중요합니다! ---
if __name__ == '__main__':
    # 윈도우 멀티프로세싱 에러를 방지하기 위해 
    # 반드시 이 안에서 실행 코드를 호출해야 합니다.
    train_model()