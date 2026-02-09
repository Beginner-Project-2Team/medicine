import os
import pandas as pd
import torch
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms

# 1. 환경 설정 및 에러 방지
# 그래픽카드(CUDA)가 사용 가능하면 GPU를 쓰고, 아니면 CPU를 쓰도록 설정합니다.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 2. 경로 설정
# 테스트용 이미지가 들어있는 폴더 경로입니다. (r을 붙여 역슬래시 에러 방지)
TEST_IMG_DIR = r'C:\Users\User\Documents\Python\project_1\pj\medicine\data\raw\test_images'
# 결과물이 저장될 경로입니다.
SUBMISSION_PATH = r"C:\Users\User\Desktop\medicine\submission.csv"

# 3. 모델 로드
#  학습시켜 저장한 '지식(가중치)' 파일인 last_model.pth를 불러옵니다.
model.load_state_dict(torch.load("last_model.pth", map_location=device))
# 모델을 GPU(또는 CPU) 메모리에 올립니다.
model.to(device)
# 모델을 '추론 모드'로 바꿉니다. (학습용 기능을 끄고 시험 모드로 전환)
model.eval()

# 4. 추론 및 결과 정리
# 폴더 내 이미지 파일들만 골라내어 이름순으로 정렬합니다.
test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

norm = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

results_list = [] # 정답 데이터를 담을 빈 리스트입니다.
ann_id_counter = 1 # 제출 양식의 고유 번호(annotation_id)를 1부터 시작합니다.

print(f"--- 추론 시작 (총 {len(test_images)}장) ---")

# 기울기 계산을 끕니다. (추론할 때는 메모리 절약을 위해 필수입니다.)
with torch.no_grad():
    for img_name in test_images:
        # 이미지의 전체 경로를 만듭니다.
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        # 파일명에서 숫자만 뽑아 image_id로 씁니다. (예: test_1.jpg -> 1)
        image_id = int(''.join(filter(str.isdigit, img_name))) 
        
        # 이미지를 열고 RGB 색상 모드로 바꿉니다.
        img = Image.open(img_path).convert("RGB")
        # 이미지를 텐서(숫자 뭉치)로 바꾸고 GPU로 보냅니다.
        img_tensor = norm(F.to_tensor(img)).to(device)
        # 모델에게 이미지를 주고 결과를 예측합니다.
        outputs = model([img_tensor])

        # 예측된 박스 좌표, 라벨(번호), 신뢰도 점수를 꺼냅니다.
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        for i in range(len(boxes)):
            # 신뢰도 점수가 0.25 미만(긴가민가한 것)은 가차 없이 버립니다.
            if scores[i] < 0.25: continue 

            # 박스의 왼쪽 위(min)와 오른쪽 아래(max) 좌표를 가져옵니다.
            x_min, y_min, x_max, y_max = boxes[i]
            
            # [핵심] 모델 번호(1, 2...)를 실제 약물 고유 ID(13900...)로 번역합니다.
            model_label = int(labels[i])
            original_category_id = model_to_catid.get(model_label, model_label)

            # 정답지 양식에 맞춰 정보를 차곡차곡 담습니다.
            results_list.append({
                'annotation_id': ann_id_counter, # 고유 번호
                'image_id': image_id,           # 이미지 번호
                'category_id': original_category_id, # 실제 약 번호
                'bbox_x': int(round(float(x_min))),  # 박스 시작 x점
                'bbox_y': int(round(float(y_min))),  # 박스 시작 y점
                'bbox_w': int(round(float(x_max - x_min))), # 박스 너비
                'bbox_h': int(round(float(y_max - y_min))), # 박스 높이
                'score': round(float(scores[i]), 2)  # 신뢰도 점수 (소수점 2자리)
            })
            ann_id_counter += 1 # 다음 정답을 위해 번호를 1 올립니다.

# 5. CSV 저장
# 모아둔 정답 리스트를 표(DataFrame) 형태로 만듭니다.
df = pd.DataFrame(results_list)
# 표를 CSV 파일로 저장합니다. (인덱스 번호는 제외)
df.to_csv(r"C:\Users\User\Desktop\submission.csv", index=False)
print("생성 완료!")