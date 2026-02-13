"""
NMS 앙상블 스크립트
두 모델의 예측 CSV를 합친 뒤 NMS로 중복 박스 제거
사용법: python ensemble/ensemble.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torchvision.ops import nms

# ============================================================
# 설정 (여기서 수정)
# ============================================================
CSV_A = "FastRCNN_1.csv"       # submit/ 안의 첫 번째 CSV
CSV_B = "yolov8m0.25.csv"      # submit/ 안의 두 번째 CSV
OUT_NAME = "ensemble_preds.csv"

IOU_THR = 0.5                  # NMS IoU threshold
SCORE_MIN = 0.2               # 최소 score (이하 제거)
# ============================================================


def run_ensemble(csv_a=CSV_A, csv_b=CSV_B, out_name=OUT_NAME,
                 iou_thr=IOU_THR, score_min=SCORE_MIN):
    submit_dir = PROJECT_ROOT / "submit"

    df_a = pd.read_csv(submit_dir / csv_a)
    df_b = pd.read_csv(submit_dir / csv_b)

    df_a["source"] = "A"
    df_b["source"] = "B"

    df_all = pd.concat([df_a, df_b], ignore_index=True)
    df_all = df_all[df_all["score"] >= score_min]

    ensemble_rows = []
    new_ann_id = 0

    for (image_id, category_id), df_grp in df_all.groupby(["image_id", "category_id"]):
        x1 = df_grp["bbox_x"]
        y1 = df_grp["bbox_y"]
        x2 = df_grp["bbox_x"] + df_grp["bbox_w"]
        y2 = df_grp["bbox_y"] + df_grp["bbox_h"]

        boxes = torch.tensor(
            list(zip(x1, y1, x2, y2)),
            dtype=torch.float32,
        )
        scores = torch.tensor(df_grp["score"].values, dtype=torch.float32)

        keep = nms(boxes, scores, iou_threshold=iou_thr)
        boxes_nms = boxes[keep]
        scores_nms = scores[keep]

        for b, s in zip(boxes_nms, scores_nms):
            new_ann_id += 1
            bx1, by1, bx2, by2 = b.tolist()

            ensemble_rows.append({
                "annotation_id": new_ann_id,
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox_x": bx1,
                "bbox_y": by1,
                "bbox_w": bx2 - bx1,
                "bbox_h": by2 - by1,
                "score": float(s),
            })

    df_ens = pd.DataFrame(ensemble_rows)
    output_path = submit_dir / out_name
    df_ens.to_csv(output_path, index=False)
    print(f"Saved NMS ensemble to {output_path}, total rows = {len(df_ens)}")


if __name__ == "__main__":
    run_ensemble()
