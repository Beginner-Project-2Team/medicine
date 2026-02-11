# pathlib: 경로를 문자열이 아니라 "객체"로 다루게 해주는 표준 라이브러리
# → OS(윈도우/맥) 달라도 안전하게 경로 처리 가능
"""
MEDICINE 프로젝트 경로 관리 모듈
paths.yml을 읽어서 모든 경로를 절대경로로 변환
"""

from pathlib import Path
import yaml

# 이 파일(config/load_paths.py)의 실제 위치를 기준으로 프로젝트 최상위 폴더(medicine/)를 자동으로 찾아냄
PROJECT_ROOT = Path(__file__).resolve().parents[1] # parents[1] → config의 한 단계 위 = 프로젝트 루트

# paths.yml 파일 읽기
PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yml"

with open(PATHS_FILE, "r", encoding="utf-8") as f: # encoding="utf-8": 한글 주석 깨짐 방지
    paths_config = yaml.safe_load(f)  # yml 내용을 Python 딕셔너리로 변환

# 상대경로를 절대경로로 변환하는 함수 -> yml에 적힌 상대경로를 실제 사용 가능한 절대경로로 변환
def get_absolute_path(relative_path):
    """yml에 적힌 상대경로를 절대경로로 변환"""
    if relative_path is None:
        return None
    return (PROJECT_ROOT / relative_path).resolve()

# 경로 변수 정의

# 데이터
DATA_PROCESSED = get_absolute_path(paths_config["data"]["processed"])
DATA_TEST_IMAGES = get_absolute_path(paths_config["data"]["test_images"])
DATA_TRAIN_ANNOTATIONS = get_absolute_path(paths_config["data"]["train_annotations"])
DATA_TRAIN_IMAGES = get_absolute_path(paths_config["data"]["train_images"])

# 데이터셋
DATASET_YOLO = get_absolute_path(paths_config["dataset"]["yolo_dataset"])

