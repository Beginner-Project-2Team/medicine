from imports import *
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"base_path : {base_path}")

# 결로 정의
paths = {
    "train_img" : os.path.join(base_path, "data", "raw", "train_images"),
    "train_anno" : os.path.join(base_path, "data", "raw", "train_annotations"),
    "test_img" : os.path.join(base_path, "data", "raw", "test_images")
}

# 개수 파악
report = {k: len(glob.glob(os.path.join(v, "**/*"), recursive=True)) for k, v in paths.items()}

print("--데이터 현황--")
for k, v in report.items():
    print(f"{k} : {v}개\n")

"""
trian_images : 231개
train_anno : 1244개
test_img : 842개
"""

#알약 클래스 확인
pill_name = []
json_paths = glob.glob(os.path.join(paths["train_anno"], "**", "*.json"), recursive=True)

for jp in json_paths:
    with open(jp, "r", encoding = "utf-8") as f:
        data = json.load(f)
        for cat in data.get("categories", []):
            pill_name.append(cat["name"])

counter = Counter(pill_name)
print("-----알약 클래스 -----")
print(f" 총 class 종류 : {len(counter)}개")
print(f"가장 많은 알약 : {counter.most_common(3)}") # top 3
print(f"가장 적은 알약 : {counter.most_common()[:-4:-1]}\n") # bottom 3



# 사진 한장당 json이 몇개식 붙어있는지 확인
ann_dir = os.path.join(base_path, "data", "raw", "train_annotations")
json_paths = glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive=True)

img_to_ann_count = Counter()
for jp in json_paths:
    with open(jp, "r", encoding ="utf =8") as f:
        data = json.load(f)
        f_name = data["images"][0]["file_name"]
        img_to_ann_count[f_name] += 1

print("--이미지당 정답지 개수 확인--\n")
print(f"가장 많은 정답지가 붙은 이미지 {img_to_ann_count.most_common(1)}")


""""
--- 클래스 불균형 문제 ---
일양 하이트린정 200mg : 153개
기넥신에프정 : 45개
아토젯정 : 37개
.....
카나브정 3개
아빌라파이정 3개
신바로정 3개

대책 계획
1. 데이터 증강 및 가중치 조절
Over-sampling: 데이터가 10개 미만인 희귀 알약들에 대해서만 
회전(Rotation), 반전(Flip), 밝기 조절 등을 적용해 강제로 숫자를 늘립니다.

Weighted Loss: 모델이 학습할 때 데이터가 적은 알약을 틀리면 더 큰 벌점을
주는 방식을 채택합니다.

Mosaic Augmentation: 여러 장의 사진을 이어 붙여 모델이 작은 객체도 잘 찾게 만듭니다.

--- 데이터 중복 문제 ---
사진은 232장인데 정답지는 1244개

대책 계획
1. file_to_info 딕셔너리를 활용하여 이미지 파일명(file_name)을 키로
2. 중복된 json 파일들을 읽들 때, 이미 존재하는 이미지 파일명이라면 박스 정보(anns)만 추가

--- 훈련 : 테스트 데이터 비율 문제 ---
train_images : 232장
test_images : 842장

대책 : 데이터 재분할
지금 제공된 test_images에 정답(JSON)이 없다면 어쩔 수 없지만, 만약 test 폴더에도
 정답지가 있다면 전체 데이터를 하나로 모은 뒤 8:1:1(훈련:검증:테스트)로 직접 다시 나눕니다.

만약 test 데이터에 정답지가 없다면, 현재 train 데이터 232장 내에서 
K-Fold Cross Validation을 사용해 데이터 부족을 기술적으로 극복해야 합니다.
"""