from imports import *
import re


def run_inference():
    # 1. 학습된 모델 로드
    model_path = r'C:\Users\KIMJW\Desktop\medicine\runs\detect\pill_project\yolov8_pill_detect4\weights\best.pt'
    model = YOLO(model_path)
    model_to_catid = {}

    for k, v in model.names.items():
        numeric_id = re.sub(r'[^0-9]', '', v) 
        model_to_catid[int(k)] = int(numeric_id)


    # 2. 테스트 이미지 경로 및 파일 목록 확보
    test_img_dir = r'C:\Users\KIMJW\Desktop\medicine\data\raw\test_images'
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    submission_data = []
    ann_id = 1
    
    print("🚀 추론 시작...")
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_img_dir, img_name)
        
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
                
    # 3. CSV 저장
    df = pd.DataFrame(submission_data)
    output_path = "submission_yolov8s_fixed.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ 제출 파일 생성 완료: {output_path}")

if __name__ == "__main__":
    run_inference()