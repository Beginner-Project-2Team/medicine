# 🧬 Pill Detection Project (AI 초급 프로젝트)

## 💊 프로젝트 소개 (Overview)

이미지 인식 기술을 이용해 사진 속 경구약제(알약)의 종류와 위치를 검출하는 객체 탐지(Object Detection) 프로젝트이다.

초기에는 Faster R-CNN을 베이스라인으로 구축해 데이터 특성과 문제 구조를 분석하고, 이후 YOLOv8 모델을 추가로 적용하여 성능과 실시간성을 비교·검증하였다.

또한 클래스 불균형을 해결하기 위해 데이터 확장(Data Expansion)을 적용하여 성능을 향상시켰다.

## 👥 팀 구성 및 역할

| 이름  | 역할                                     | 주요 업무                                            |
| --- | -------------------------------------- | ------------------------------------------------ |
| 김예주 | Project Manager / Experimentation Lead | 일정 관리, 협업 진행, 전체 방향 조율, 실험 관리, 하이퍼파라미터 튜닝, 성능 평가 |
| 박윤민 | Data Engineer                          | 데이터 구조 파악, EDA, 전처리                              |
| 최지훈 | Data Engineer / Experimentation Lead   | 데이터 구조 파악, EDA, 전처리, 실험 관리, 하이퍼파라미터 튜닝, 성능 평가    |
| 임운하 | Model Architect (Faster R-CNN)         | 모델 선정 및 구조 설계                                    |
| 김정우 | Model Architect (YOLOv8)               | 모델 선정 및 구조 설계                                    |

> ※ 역할은 담당 기준으로 구분되어 있으나, 모든 과정은 팀원 전원이 함께 논의하고 공동으로 수행했습니다.




## 💊 환경 설정 (Environment Setup)

본 프로젝트는 Conda 기반 Python 3.13 환경을 기준으로 구성됨, GPU 사용은 선택 사항이며, GPU가 없는 환경에서도 실행할 수 있다.

environment.yml이 공식 실행 환경이며, requirements.txt는 보조 설치용이다.

⚠️ requirements.txt는 최소 실행용 의존성만 포함되어 있으며 완전한 실험 재현은 보장되지 않을 수 있음

## 💊 Dataset & Pretrained Weights

데이터 용량 문제로 raw / processed 데이터는 GitHub에 포함되어 있지 않습니다.
아래 Google Drive에서 다운로드 후 지정 경로에 배치해주세요.

>train_annotations_수정 파일은 다운로드 후 파일명에서 _수정을 제거하여
train_annotations 로 변경해야 정상 동작합니다.

👉 https://drive.google.com/drive/folders/1d0beobFkyemEEgCDoHOulmUiKxZ0gr8g?usp=sharing 

이미지 병합 데이터는 제공되지 않으며,
notebooks/image_merge.ipynb 실행 시 자동 생성됩니다.

## 💊 폴더 구조 (Project Structure)

```
medicine/
├── configs/            # 경로 설정 (paths.yml, load_paths.py)
├── preprocessing/      # 데이터 전처리 (COCO 통합, YOLO 변환, 품질 검사)
├── dataset/            # 데이터셋 클래스 (RCNN Dataset, YOLO data.yaml)
├── model/
│   ├── faster_rcnn/    # Faster R-CNN (모델 정의, 학습, 추론)
│   └── yolov8s/        # YOLOv8 (학습, 추론)
├── runner/             # 통합 실행 스크립트 (main_yolo.py, main_rcnn.py)
├── ensemble/           # NMS 앙상블
├── notebooks/          # 탐색/실험용 노트북
├── data/               # 원본/전처리 데이터 (gitignore)
├── outputs/            # 학습 결과 저장 (gitignore)
└── submit/             # 제출 CSV 저장
```

## 💊 실행 방법 (Usage)

### 1. 데이터 증강 (선택)
증강 데이터를 사용하려면 `notebooks/image_merge.ipynb`를 먼저 실행하여 병합 이미지를 생성합니다.

### 2. 학습 + 추론
```bash
# YOLOv8 (데이터 변환 → 학습 → 추론)
python runner/main_yolo.py

# Faster R-CNN (학습 → 추론)
python runner/main_rcnn.py
```
각 runner 파일 상단의 하이퍼파라미터와 `DO_TRAIN`, `DO_INFERENCE` 등을 수정하여 원하는 단계만 실행할 수 있습니다.

각 스크립트는 개별 실행도 가능합니다.
```bash
python preprocessing/yolo_converter.py         # COCO → YOLO 변환만
python model/yolov8s/yolov8s_model.py          # YOLO 학습만
python model/yolov8s/yolo_inference.py         # YOLO 추론만
python model/faster_rcnn/rcnn_train.py         # RCNN 학습만
python model/faster_rcnn/rcnn_inference.py     # RCNN 추론만
```

### 3. 앙상블 (선택)
```bash
python ensemble/ensemble.py
```
YOLO와 RCNN 예측 CSV 2개를 NMS로 병합합니다. 상단에서 CSV 파일명을 지정합니다.
- 자세한 내용은 notebooks/PROJECT_STRUCTURE.ipynb에서 확인할 수 있습니다.


## 🤝 협업 일지
김예주 - https://www.notion.so/2fa431ae37e5802393b1d5eef42b7b0e?source=copy_link

김정우 - https://www.notion.so/2fdb37fe0bdd8005aa71d5bd49f38069?v=2fdb37fe0bdd804eaa54000c95ae5315

박윤민 - https://www.notion.so/2-Daily-2f7dc68f96af802ebf18de92a2913a02

임운하 - 업무일지 폴더에 pdf파일 있음

최지훈 - https://www.notion.so/306f1a760e2780eb94fde9e3fdfa6c9e?source=copy_link

## 📄 최종 보고서

👉 https://jet-port-669.notion.site/2-3030b7469455809fa661e57a5c9c548c?source=copy_link