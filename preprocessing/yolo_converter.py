from imports import *
from configs.load_paths import DATA_TRAIN_IMAGES, DATA_TRAIN_ANNOTATIONS, DATASET_YOLO
from preprocessing.yolo_get_intergrated_data import yolo_get_integrated_data
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

    def run(self):
        for mode in ['train', 'val']:
            (self.yolo_root / "images" / mode).mkdir(parents=True, exist_ok=True)
            (self.yolo_root / "labels" / mode).mkdir(parents=True, exist_ok=True)

        print("📦 데이터를 통합 중...")
        master_data = yolo_get_integrated_data(self.raw_ann_dir)
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

        train_imgs, val_imgs = train_test_split(all_imgs, train_size=self.train_ratio, random_state=42)

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
        print(f"\n✅ 완료! 위치: {self.yolo_root}")

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
    converter.run()