"""
Faster R-CNN 추론 및 제출 CSV 생성
사용법: python model/faster_rcnn/rcnn_inference.py
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from configs.load_paths import DATA_TEST_IMAGES, DATA_TRAIN_ANNOTATIONS
from model.faster_rcnn.rcnn_model import create_model
from model.faster_rcnn.rcnn_train import build_class_mapping

# ============================================================
# 하이퍼파라미터 (여기서 수정)
# ============================================================
CONF_THRESHOLD = 0.25      # confidence threshold (낮출수록 검출 많아짐)
MODEL_PATH = None           # None이면 outputs/rcnn/ 에서 최신 best 자동 탐색
OUTPUT_NAME = None          # None이면 모델 이름으로 자동 생성 (예: rcnn1.csv)
# ============================================================


def rcnn_inference(
    model_path=MODEL_PATH,
    conf_threshold=CONF_THRESHOLD,
    output_name=OUTPUT_NAME,
):
    """
    학습된 Faster R-CNN 모델로 테스트 이미지 추론 후 제출 CSV 생성
    """
    # 1. 클래스 매핑 로드
    _, model_to_catid, num_classes = build_class_mapping(
        DATA_TRAIN_ANNOTATIONS
    )

    # 2. 모델 로드 (outputs/rcnn/ 에서 최신 best 모델 자동 탐색)
    if model_path is None:
        rcnn_dir = PROJECT_ROOT / "outputs" / "rcnn"
        if rcnn_dir.exists():
            # rcnn1.pth, rcnn2.pth 등 (_last 제외)
            best_files = sorted(
                [f for f in rcnn_dir.glob("*.pth") if "_last" not in f.name],
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            if best_files:
                model_path = best_files[0]
        if model_path is None:
            print(f"모델 파일을 찾을 수 없습니다!")
            print(f"   확인 경로: {rcnn_dir}")
            print(f"   먼저 rcnn_train.py로 학습을 실행하세요.")
            return

    print(f"모델 로딩: {model_path}")
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 테스트 이미지 목록
    test_img_dir = DATA_TEST_IMAGES
    test_images = sorted([
        f.name for f in test_img_dir.glob("*")
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    # 4. 정규화 (학습 시와 동일)
    norm = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # 5. 추론
    results_list = []
    ann_id_counter = 1

    print(f"추론 시작 (총 {len(test_images)}장, conf >= {conf_threshold})")

    with torch.no_grad():
        for img_name in tqdm(test_images):
            img_path = test_img_dir / img_name
            image_id = int(''.join(filter(str.isdigit, img_name)))

            img = Image.open(img_path).convert("RGB")
            # PIL → Tensor → float [0,1] → 정규화 (학습 Transform과 동일)
            img_tensor = F.pil_to_tensor(img)
            img_tensor = F.to_dtype(img_tensor, dtype=torch.float32, scale=True)
            img_tensor = norm(img_tensor).to(device)

            outputs = model([img_tensor])

            boxes = outputs[0]['boxes'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()

            for i in range(len(boxes)):
                if scores[i] < conf_threshold:
                    continue

                x_min, y_min, x_max, y_max = boxes[i]
                w = x_max - x_min
                h = y_max - y_min

                if w <= 0 or h <= 0:
                    continue

                # 모델 label → 원본 category_id로 변환
                model_label = int(labels[i])
                original_category_id = model_to_catid.get(model_label, model_label)

                results_list.append({
                    'annotation_id': ann_id_counter,
                    'image_id': image_id,
                    'category_id': original_category_id,
                    'bbox_x': int(round(float(x_min))),
                    'bbox_y': int(round(float(y_min))),
                    'bbox_w': int(round(float(w))),
                    'bbox_h': int(round(float(h))),
                    'score': round(float(scores[i]), 4)
                })
                ann_id_counter += 1

    # 6. CSV 저장 (모델 이름으로 자동 생성)
    df = pd.DataFrame(results_list)
    output_dir = PROJECT_ROOT / "submit"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        # 모델 경로에서 이름 추출 (예: outputs/rcnn/rcnn1.pth → rcnn1.csv)
        output_name = Path(model_path).stem + ".csv"
    output_path = output_dir / output_name
    df.to_csv(output_path, index=False)
    print(f"제출 파일 생성 완료: {output_path}")
    print(f"총 {len(results_list)}개 검출 ({len(test_images)}장 이미지)")


if __name__ == "__main__":
    rcnn_inference()
