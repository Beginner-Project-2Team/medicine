# 🩺 Pill Detection Project (AI 초급 프로젝트)

이미지 인식 기술을 활용하여  
**사진 속 경구약제(알약)의 이름과 위치를 검출하는 객체 인식 모델**을 개발하는 팀 프로젝트입니다.

---

## 📌 프로젝트 개요

- 헬스케어 스타트업 **Health Eat**의 AI 엔지니어링 팀이라는 가정 하에 진행
- 모바일 앱으로 촬영한 약 사진을 기반으로  
  → **최대 4개의 알약 클래스 + 바운딩 박스**를 예측
- Kaggle Private Competition을 통해 성능 검증 진행
- **순위보다 성능 개선 과정과 협업 경험을 중시**

---

## 🎯 프로젝트 목표

- 객체 인식(Object Detection) 파이프라인 이해 및 구현
- 데이터 전처리 → 모델 설계 → 실험 → 성능 개선의 전체 흐름 경험
- GitHub 기반 협업 경험 및 프로젝트 결과물 정리

---

## 🗓 프로젝트 일정

- **프로젝트 기간**: OT 이후 ~ 프로젝트 종료일
- **Github / 보고서 제출 마감**: 종료일 D-1 19:00 (주말 제외)
- **협업일지 제출 마감**: 프로젝트 종료일 23:50
- **Kaggle 마감**: 프로젝트 종료일 23:50

---

## 👥 팀 구성 및 역할

| 역할 | 주요 업무 |
| --- | --- |
| Project Manager | 일정 관리, 협업 진행, 전체 방향 조율 |
| Data Engineer | 데이터 구조 파악, EDA, 전처리 |
| Model Architect | 모델 선정 및 구조 설계 |
| Experimentation Lead | 실험 관리, 하이퍼파라미터 튜닝, 성능 평가 |

> 역할은 고정되지 않으며, 필요에 따라 유연하게 조정합니다.

## Data Directory Structure

본 프로젝트는 데이터 용량 및 관리 문제로 인해  
**원본 데이터(raw)와 전처리된 데이터(processed)를 GitHub에 포함하지 않습니다.**

# 📦 Environment Setup

이 프로젝트는 Python 3.13과 conda 환경을 기준으로 합니다.
GPU 사용은 선택 사항이며, GPU가 없는 환경에서도 실행할 수 있습니다.

1️⃣ Conda 환경 생성 (권장)

conda env create -f environment.yml
conda activate medicine


Python 3.13 기반 환경이 생성됩니다.

데이터 분석 및 기본 실행에 필요한 패키지가 설치됩니다.

2️⃣ pip만 사용하는 경우 (대안)

conda를 사용하지 않는 경우 아래 명령으로 최소 실행 환경을 구성할 수 있습니다.

pip install -r requirements.txt


CPU 기준으로 설치됩니다.

GPU가 없어도 코드 실행이 가능합니다.

📁 Path 규칙

프로젝트 루트: medicine/

노트북:

PROJECT_ROOT = Path.cwd().parent


스크립트:

PROJECT_ROOT = Path(__file__).resolve().parents[2]


모든 경로는 PROJECT_ROOT 기준으로 작성