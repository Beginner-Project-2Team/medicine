"""
Faster R-CNN용 Dataset 클래스
COCO 포맷 데이터를 Faster R-CNN이 이해할 수 있는 형태로 변환
사용법: python dataset/faster_rcnn/rcnn_dataset.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
from configs.load_paths import DATA_TRAIN_IMAGES, DATA_TEST_IMAGES
from preprocessing.coco_data_integration import get_integrated_coco_data


class PillDataset(Dataset):
    def __init__(self, split="train", transforms=None, catid_to_model=None):
        self.split = split
        self.transforms = transforms
        self.catid_to_model = catid_to_model or {}

        if split in ("train", "val"):
            # [핵심] 공통 데이터 통합 함수 사용 (YOLO와 동일한 데이터!)
            self.integrated_data = get_integrated_coco_data()
            all_filenames = sorted(self.integrated_data.keys())

            # [계층화 분할] 클래스 비율을 유지하면서 train/val 나누기
            # 각 이미지의 대표 클래스(가장 많이 등장하는 클래스)를 기준으로 분할
            # → 샘플이 적은 클래스도 train/val에 비율대로 배분됨
            primary_labels = []
            class_counts = Counter()
            for fn in all_filenames:
                labels = self.integrated_data[fn]["labels"]
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

            train_files, val_files = train_test_split(
                all_filenames, train_size=0.8, random_state=42,
                stratify=stratify_labels
            )

            self.filenames = train_files if split == "train" else val_files
            self.img_dir = DATA_TRAIN_IMAGES
            print(f"  {split} split: {len(self.filenames)}개 이미지")

        elif split == "test":
            # 테스트는 이미지만 있음 (정답 없음)
            self.img_dir = DATA_TEST_IMAGES
            self.filenames = sorted([
                f.name for f in DATA_TEST_IMAGES.glob("*")
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
            ])
            self.integrated_data = None

        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, file_name: str):

        img_path = self.img_dir / file_name
        img = Image.open(img_path).convert("RGB")
        return img

    def _load_target(self, file_name: str, image_id: int):
   
        # 통합 데이터에서 해당 이미지 정보 가져오기
        info = self.integrated_data[file_name]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        # 각 박스마다 처리
        for box, orig_label in zip(info['boxes'], info['labels']):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            # 크기 검증 (0 이하면 무시)
            if w <= 0 or h <= 0:
                continue

            # [중요] 원본 category_id를 모델용 label로 변환
            # 예: 원본 ID 13900123 → 모델 label 1
            if orig_label not in self.catid_to_model:
                continue  # 매핑 안 되는 클래스는 무시

            model_label = self.catid_to_model[orig_label]

            boxes.append([x1, y1, x2, y2])
            labels.append(model_label)
            areas.append(w * h)
            iscrowd.append(0)  # 우리 데이터는 군중 아님

        # 리스트를 텐서로 변환
        if len(boxes) == 0:
            # 박스가 없으면 빈 텐서 생성
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Faster R-CNN이 요구하는 딕셔너리 형태로 반환
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return target

    def __getitem__(self, idx):
     
        file_name = self.filenames[idx]
        img = self._load_image(file_name)

        image_id = idx  # 학습 내부용 ID

        if self.split in ("train", "val"):
            target = self._load_target(file_name, image_id)
        else:
            # 테스트는 image_id만
            target = {"image_id": torch.tensor([image_id])}

        # PIL Image → Tensor 변환
        img = TVImage(F.pil_to_tensor(img))  # [C, H, W], uint8

        if self.split in ("train", "val"):
            # BoundingBoxes 객체로 변환 (torchvision v2 형식)
            target["boxes"] = BoundingBoxes(
                target["boxes"],
                format="XYXY",  # (x1, y1, x2, y2) 형식
                canvas_size=F.get_size(img)
            )

        # 데이터 증강/정규화 적용 (있으면)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # [안전장치] 학습/검증 데이터는 유효한 박스만 남기기
        if self.split in ("train", "val"):
            boxes = target["boxes"]  # [N, 4]
            if boxes.numel() > 0:
                x1, y1, x2, y2 = boxes.unbind(dim=1)
                # x2 > x1 이고 y2 > y1 인 박스만 유지
                keep = (x2 > x1) & (y2 > y1)

                target["boxes"] = boxes[keep]
                target["labels"] = target["labels"][keep]
                target["area"] = target["area"][keep]
                target["iscrowd"] = target["iscrowd"][keep]

        return img, target


# [중요] Detection용 collate 함수
def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
