"""
Faster R-CNN 학습 스크립트
my.ipynb의 학습 코드를 Python 파일로 정리한 버전
공통 데이터 통합 함수(get_integrated_coco_data)를 활용
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (import 경로 해결)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from imports import *
import torchvision.transforms.v2 as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from configs.load_paths import DATA_TRAIN_ANNOTATIONS
from dataset.faster_rcnn.rcnn_dataset import PillDataset, detection_collate_fn
from model.faster_rcnn.rcnn_model import create_model

# 1. 클래스 매핑 구축
# [핵심] 원본 category_id → 모델 내부 label로 변환하는 매핑

def collect_unique_cats(ann_dir):
    cat_ids = set()
    for dirpath, _, filenames in os.walk(ann_dir):
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            jp = os.path.join(dirpath, fn)
            with open(jp, "r", encoding="utf-8") as f:
                coco = json.load(f)
            for ann in coco["annotations"]:
                cat_ids.add(ann["category_id"])
    return sorted(list(cat_ids))


def build_class_mapping(ann_dir):
    unique_cats = collect_unique_cats(ann_dir)

    # background = 0, 나머지 클래스는 1~K
    catid_to_model = {cat_id: i + 1 for i, cat_id in enumerate(unique_cats)}
    model_to_catid = {v: k for k, v in catid_to_model.items()}

    num_classes = 1 + len(unique_cats)  # 1(배경) + 56(알약) = 57
    print(f"NUM_CLASSES: {num_classes}")

    return catid_to_model, model_to_catid, num_classes


# 2. Transform 정의
# [참고] torchvision.transforms.v2를 사용
# 학습용 Transform
train_transform = T.Compose([
    # uint8 [0,255] → float32 [0,1]로 스케일링
    T.ToDtype(torch.float32, scale=True),

    # --- 데이터 증강 (필요시 주석 해제) ---
    # T.RandomHorizontalFlip(p=0.5),           # 좌우 뒤집기
    # T.RandomVerticalFlip(p=0.2),             # 상하 뒤집기
    # T.RandomApply([
    #     T.RandomAffine(
    #         degrees=5,                        # 회전 (±5도)
    #         translate=(0.02, 0.02),            # 이동
    #         scale=(0.9, 1.3),                  # 크기 변환
    #         shear=5,                           # 기울이기
    # )], p=0.5),
    # T.RandomApply([
    #     T.ColorJitter(
    #         brightness=0.2,                    # 밝기
    #         contrast=0.2,                      # 대비
    #         saturation=0.1,                    # 채도
    #         hue=0.02,                          # 색조
    #     )
    # ], p=0.5),

    # ImageNet 정규화 (사전학습 모델이 이 값으로 학습되었으므로 맞춰야 함)
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

# 테스트/추론용 Transform (증강 없음, 정규화만)
test_transform = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

# 3. 학습 함수들

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    # model.train()을 해야 학습 모드로 전환됨
    # (Dropout, BatchNorm 등이 학습 모드로 동작)
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # 이미지와 타겟을 GPU로 이동
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in t.items()} for t in targets]

        # [핵심] Faster R-CNN은 train 모드에서 (images, targets) 넣으면 loss dict 반환
        loss_dict = model(images, targets)

        # 4개 loss를 합산
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        running_loss += loss_value

        # 역전파 3단계: 초기화 → 역전파 → 업데이트
        optimizer.zero_grad()  # 이전 기울기 초기화
        losses.backward()      # 기울기 계산
        optimizer.step()       # 가중치 업데이트

    return running_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate_mAP(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    # [추가] mAP@[0.75:0.95] 계산용 (캐글 대회 지표)
    metric_75_95 = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.75, 0.80, 0.85, 0.90, 0.95]
    )

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]

        # [핵심] eval 모드에서는 예측 결과를 반환 (loss가 아님!)
        outputs = model(images)

        preds = []
        gts = []
        for output, target in zip(outputs, targets):
            preds.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
            gts.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

        metric.update(preds, gts)
        metric_75_95.update(preds, gts)

    result = metric.compute()
    result_75_95 = metric_75_95.compute()
    return {
        "mAP": result["map"].item(),              # mAP@[0.5:0.95] (COCO 기본)
        "mAP_50": result["map_50"].item(),         # mAP@IoU=0.5
        "mAP_75": result["map_75"].item(),         # mAP@IoU=0.75
        "mAP_75_95": result_75_95["map"].item(),   # [추가] mAP@[0.75:0.95] (캐글 지표)
    }


def train_only(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    device,
    num_epochs=10,
    print_freq=50,
    save_path="last_model.pth",
):
    best_mAP = 0.0

    for epoch in range(1, num_epochs + 1):
        # --- 학습 ---
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=print_freq
        )

        # --- 검증 (mAP 측정) ---
        mAP_result = evaluate_mAP(model, val_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"train_loss: {train_loss:.4f} | "
            f"mAP@50: {mAP_result['mAP_50']:.4f} | "
            f"mAP@[.5:.95]: {mAP_result['mAP']:.4f} | "
            f"mAP@[.75:.95]: {mAP_result['mAP_75_95']:.4f}"  # [추가] 캐글 지표
        )

        # best mAP 갱신 시 모델 저장
        if mAP_result["mAP_50"] > best_mAP:
            best_mAP = mAP_result["mAP_50"]
            best_path = save_path.replace("_last.pth", ".pth")
            torch.save(model.state_dict(), best_path)
            print(f"  >> Best model saved! (mAP@50: {best_mAP:.4f})")

        # [중요] 스케줄러 업데이트 (에포크마다 학습률 조정)
        lr_scheduler.step()

    # 마지막 에포크 모델도 저장
    torch.save(model.state_dict(), save_path)
    print(f"Saved last model to {save_path}")


# 4. 메인 실행

def run_training(batch_size=4, num_epochs=2, lr=0.01, name="rcnn1"):
    """Faster R-CNN 학습 파이프라인"""
    # 모델 저장 경로 (outputs/rcnn/rcnn1.pth)
    save_dir = PROJECT_ROOT / "outputs" / "rcnn"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{name}_last.pth"

    # 클래스 매핑
    catid_to_model, _, num_classes = build_class_mapping(
        DATA_TRAIN_ANNOTATIONS
    )

    # 데이터셋 생성
    train_dataset = PillDataset(
        split="train",
        transforms=train_transform,
        catid_to_model=catid_to_model,
    )

    val_dataset = PillDataset(
        split="val",
        transforms=test_transform,
        catid_to_model=catid_to_model,
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    # 모델 생성
    model = create_model(num_classes)
    model.to(device)

    # 옵티마이저 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=11,
        gamma=0.5,
    )

    # 학습 시작
    print(f"\n{'='*50}")
    print(f"Faster R-CNN 학습 시작")
    print(f"NUM_CLASSES: {num_classes}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"NUM_EPOCHS: {num_epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Save Path: {save_path}")
    print(f"{'='*50}\n")

    train_only(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=num_epochs,
        save_path=str(save_path),
    )

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    run_training(name="rcnn1")  # ← 개별 실행 시 여기서 이름 변경
