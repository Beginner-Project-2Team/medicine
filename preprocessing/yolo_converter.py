"""
YOLO 데이터셋 변환기 (COCO → YOLO 포맷)
사용법: python preprocessing/yolo_converter.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from configs.load_paths import DATA_TRAIN_IMAGES, DATA_TRAIN_ANNOTATIONS, DATASET_YOLO
from preprocessing.coco_data_integration import get_integrated_coco_data

class PillYOLOConverter:
    def __init__(self, train_ratio=0.8):
        self.train_ratio = train_ratio
        # 공통 경로 사용
        self.raw_img_dir = DATA_TRAIN_IMAGES
        self.raw_ann_dir = DATA_TRAIN_ANNOTATIONS
        self.yolo_root = DATASET_YOLO  # dataset/yolo_dataset/

    def _convert_to_yolo(self, box, img_w, img_h):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return (cx / img_w, cy / img_h, w / img_w, h / img_h)

    def split_check(self):
        """데이터셋 생성 및 Split 관리 (덮어쓰기 방지 + Split 정보 저장)"""
        split_info_file = self.yolo_root / "split_info.json"

        # [방법 1] 이미 데이터셋이 존재하는지 확인
        if split_info_file.exists():
            print("\n⚠️  YOLO 데이터셋이 이미 존재합니다!")
            print(f"📂 위치: {self.yolo_root}")

            # 기존 split 정보 로드
            with open(split_info_file, "r", encoding="utf-8") as f:
                old_split = json.load(f)
            print(f"📊 기존 Split - Train: {old_split['train_size']}장, Val: {old_split['val_size']}장")
            print(f"🕐 생성 시각: {old_split.get('created_at', 'N/A')}")

            print("\n선택지:")
            print("  1️⃣  기존 데이터셋 재사용 (추천)")
            print("  2️⃣  삭제 후 새로 생성 (덮어쓰기)")
            print("  3️⃣  취소")

            choice = input("\n선택 [1/2/3]: ").strip()

            if choice == "1":
                print("✅ 기존 데이터셋을 재사용합니다.")
                return
            elif choice == "3":
                print("❌ 변환 작업을 취소합니다.")
                return
            elif choice == "2":
                print("🗑️  기존 데이터 삭제 후 재생성합니다...")
                shutil.rmtree(self.yolo_root)
            else:
                print("❌ 잘못된 입력입니다. 취소합니다.")
                return

        # 폴더 생성
        for mode in ['train', 'val']:
            (self.yolo_root / "images" / mode).mkdir(parents=True, exist_ok=True)
            (self.yolo_root / "labels" / mode).mkdir(parents=True, exist_ok=True)

        print("\n📦 데이터를 통합 중...")
        master_data = get_integrated_coco_data(self.raw_ann_dir)
        print(f"=====master_data : {len(master_data)}")
        all_imgs = list(master_data.keys())

        # --- [추가] 클래스 매핑 테이블 생성 ---
        # 모든 데이터에서 고유한 라벨을 뽑아 0부터 번호를 새로 매깁니다.
        all_labels = set()
        for data in master_data.values():
            all_labels.update(data['labels'])

        sorted_labels = sorted(list(all_labels))
        # {원본ID: 0, 원본ID2: 1, ...} 형태의 딕셔너리 생성
        label_map = {orig_id: i for i, orig_id in enumerate(sorted_labels)}
        print(f"🔍 총 {len(sorted_labels)}개의 클래스를 발견했습니다.")
        # ---------------------------------------

        # [계층화 분할] 클래스 비율을 유지하면서 train/val 나누기
        # RCNN과 동일한 방식: 각 이미지의 대표 클래스(가장 많이 등장하는 클래스) 기준
        primary_labels = []
        class_counts = Counter()
        for img_name in all_imgs:
            labels = master_data[img_name]['labels']
            if labels:
                primary = Counter(labels).most_common(1)[0][0]
            else:
                primary = -1  # 박스 없는 이미지
            class_counts[primary] += 1
            primary_labels.append(primary)

        # 샘플이 1개뿐인 클래스는 stratify 불가 → "rare"로 묶기
        stratify_labels = []
        for label in primary_labels:
            if class_counts[label] < 2:
                stratify_labels.append("rare")
            else:
                stratify_labels.append(str(label))

        train_imgs, val_imgs = train_test_split(
            all_imgs, train_size=self.train_ratio, random_state=42,
            stratify=stratify_labels
        )

        for mode, target_list in [('train', train_imgs), ('val', val_imgs)]:
            print(f"✍️ {mode} 데이터 생성 중...")
            for img_name in tqdm(target_list):
                data = master_data[img_name]
                img_w, img_h = data['width'], data['height']

                shutil.copy(
                    self.raw_img_dir / img_name,
                    self.yolo_root / "images" / mode / img_name)

                label_name = os.path.splitext(img_name)[0] + ".txt"
                with open(self.yolo_root / 'labels' / mode / label_name, "w") as f:
                    for box, orig_label in zip(data['boxes'], data['labels']):
                        yolo_box = self._convert_to_yolo(box, img_w, img_h)
                        # [중요] 원본 라벨 대신 매핑된 0~N 번호를 사용합니다.
                        new_label_id = label_map[orig_label]
                        f.write(f"{new_label_id} {' '.join([f'{x:.6f}' for x in yolo_box])}\n")

        # 6. data.yaml 파일 생성 (정리된 sorted_labels 전달)
        self._create_yaml(sorted_labels)

        # [방법 2] Split 정보 저장
        from datetime import datetime
        split_info = {
            "train": train_imgs,
            "val": val_imgs,
            "train_size": len(train_imgs),
            "val_size": len(val_imgs),
            "train_ratio": self.train_ratio,
            "random_state": 42,
            "stratified": True,
            "total_images": len(all_imgs),
            "num_classes": len(sorted_labels),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        split_info_file = self.yolo_root / "split_info.json"
        with open(split_info_file, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 완료! 위치: {self.yolo_root}")
        print(f"💾 Split 정보 저장: {split_info_file}")

    def _create_yaml(self, sorted_labels):
        # YOLOv8 공식 형식을 따르기 위해 nc를 명시해주는 것이 좋습니다.
        yaml_content = {
            'path': str(self.yolo_root),  # Path 객체를 문자열로 변환
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(sorted_labels), # 클래스 개수 명시
            'names': {i: f"pill_{orig_id}" for i, orig_id in enumerate(sorted_labels)}
        }
        with open(self.yolo_root / "data.yaml", "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

if __name__ == "__main__":
    converter = PillYOLOConverter()
    converter.split_check()