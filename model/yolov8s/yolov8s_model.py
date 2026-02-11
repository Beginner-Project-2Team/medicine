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

# def train_model(epochs=2, imgsz=640, batch=4, name="yolo1"):
#     # 1. 모델 로드
#     model = YOLO('yolov8s.pt')

#     # 2. 데이터셋 경로 설정 (공통 경로 사용)
#     data_yaml_path = DATASET_YOLO / "data.yaml"

#     # 3. 모델 학습 (Training)
#     results = model.train(
#         data=data_yaml_path,
#         epochs=epochs,
#         imgsz=imgsz,
#         batch=batch,
#         device=0,
#         project=str(PROJECT_ROOT / "outputs" / "yolo"),
#         name=name,
#     )

#     # 4. 성능 검증
#     metrics = model.val()
#     print(f"Mean Average Precision (mAP): {metrics.box.map}")

# # --- 이 부분이 가장 중요합니다! ---
# if __name__ == '__main__':
#     # 윈도우 멀티프로세싱 에러를 방지하기 위해
#     # 반드시 이 안에서 실행 코드를 호출해야 합니다.
#     train_model(name="yolo1")  # ← 개별 실행 시 여기서 이름 변경


#---------------------------------------------------------------------------
model = YOLO('yolov8s.pt')
def train_model(model_obj):
    # 2. 데이터셋 경로 설정
    data_yaml_path = DATASET_YOLO / "data.yaml"

    # 3. 모델 학습 (Training)
    results = model_obj.train(
        data=str(data_yaml_path),   
        epochs=1,             
        imgsz=1024,             
        batch=2,              
        device=0,              # CUDA 사용을 위해 0으로 설정
        project='pill_project', 
        name=f'yolov8_pill_{datetime.now().strftime("%Y%m%d_%H_%M")}',
        # lr0 = 0.01
        mosaic=1.0,       # 모자이크 
        mixup=0.0,        # 이미지 겹치기 
        hsv_h=0.005, hsv_s=0.2, hsv_v=0.2, # 색조 변경, 채도 변경, 명도 변경            
        degrees=30.0, flipud=0.5, fliplr=0.5,  # 회전, 상하 반전, 좌우 반전  
        translate=0.1, scale=0.2, shear=0.0,   # 이동, 확대/축소, 전단 변형 
        perspective=0.001,  # 투영 변형 
    )

    # 4. 성능 검증
    metrics = model_obj.val()
    print(f"Mean Average Precision (mAP): {metrics.box.map}")

    return metrics



def log_save(model_obj, save_dir, custom_notes="", metrics = None):
    """
    학습 조건을 txt 파일로 저장
    """
    save_dir = Path(save_dir) / "logs"
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = save_dir / f"{timestamp}.txt"

    if hasattr(model_obj, 'trainer') and model_obj.trainer is not None:
        actual_params = model_obj.trainer.args
    else:
        actual_params = model_obj.args

    #model.arg는 실제 학습 시 적용된 모든 파라미터를 담고 있는 딕셔너리 형태
    # all_args = vars(model_obj.args)

    with open(log_path, 'w', encoding = 'utf-8') as f:
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Custom Notes: {custom_notes}\n")
        f.write("-" * 50  + "\n")

        import_keys = ['model', 'data', 'epochs', 
                       'imgsz', 'batch', 'optimizer', 
                       'lr0', 'device', 'mosaic', 'mixup',
                       'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
                       'scale', 'shear', 'perspective', 'flipud', 'fliplr'
                       ]

        for key in import_keys:
            # 딕셔너리 형태일 수도, 객체 형태일 수도 있으므로 두 방식 모두 대응합니다.
            if isinstance(actual_params, dict):
                val = actual_params.get(key, 'N/A')
            else:
                val = getattr(actual_params, key, 'N/A')
            
            f.write(f"{key.upper()} : {val}\n")
        
        if metrics is not None:
            f.write("-" * 50 + "\n")
            f.write(f"RESULT mAP50 : {metrics.box.map50:.4f}\n")
            f.write(f"RESULT mAP50-95 : {metrics.box.map:.4f}\n")
        
        f.write("-" * 50 + "\n")
        f.write("Full Config captured from model trainer args.\n")
    
    print(f"log save")
 

# --- 이 부분이 가장 중요합니다! ---
if __name__ == '__main__':
    # 윈도우 멀티프로세싱 에러를 방지하기 위해 
    # 반드시 이 안에서 실행 코드를 호출해야 합니다.
    results_metrics = train_model(model)
    log_folder = PROJECT_ROOT / "outputs" / "logs"
    log_save(model_obj = model, save_dir= log_folder, metrics=results_metrics)