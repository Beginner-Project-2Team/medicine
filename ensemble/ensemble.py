import pandas as pd
import torch
from torchvision.ops import nms

"""
실행 안 되면 바로 아래 부분 수정해서 써주세요~
잘 될지도...
"""
csv_1 = pd.read_csv(r"./submission.csv")
csv_2 = pd.read_csv(r"./submit/yolov8_1.csv")
OUT_CSV = "ensemble_preds.csv"

IOU_THR = 0.5
SCORE_MIN = 0.0   # 너무 낮은 score 박스 미리 제거하고 싶으면 조절

cols = ["annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]

df_a = csv_1.copy()
df_b = csv_2.copy()

# annotation_id는 새로 매길 거라면 지금은 의미 없음
df_a["source"] = "A"
df_b["source"] = "B"

df_all = pd.concat([df_a, df_b], ignore_index=True)

# score 너무 낮은 건 미리 제거
df_all = df_all[df_all["score"] >= SCORE_MIN]


ensemble_rows = []
new_ann_id = 0

# image_id, category_id 조합으로 그룹핑
for (image_id, category_id), df_grp in df_all.groupby(["image_id", "category_id"]):

    # boxes: [N, 4] (xyxy), scores: [N]
    x1 = df_grp["bbox_x"]
    y1 = df_grp["bbox_y"]
    x2 = df_grp["bbox_x"] + df_grp["bbox_w"]
    y2 = df_grp["bbox_y"] + df_grp["bbox_h"]

    boxes = torch.tensor(
        list(zip(x1, y1, x2, y2)),
        dtype=torch.float32,
    )
    scores = torch.tensor(df_grp["score"].values, dtype=torch.float32)

    # NMS 실행
    keep = nms(boxes, scores, iou_threshold=IOU_THR)

    boxes_nms = boxes[keep]
    scores_nms = scores[keep]

    for b, s in zip(boxes_nms, scores_nms):
        new_ann_id += 1
        x1, y1, x2, y2 = b.tolist()
        w = x2 - x1
        h = y2 - y1

        ensemble_rows.append({
            "annotation_id": new_ann_id,
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox_x": x1,
            "bbox_y": y1,
            "bbox_w": w,
            "bbox_h": h,
            "score": float(s),
        })

# =========================
# 3. CSV 저장
# =========================
df_ens = pd.DataFrame(ensemble_rows)
df_ens.to_csv(OUT_CSV, index=False)

print(f"Saved NMS ensemble to {OUT_CSV}, total rows = {len(df_ens)}")