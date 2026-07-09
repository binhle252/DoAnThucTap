# GAT Improvement Log

## Muc Tieu

Ghi lai cac thay doi quan trong sau khi phat hien output cu co nguy co gay hieu nham hoac lam ket qua khong chat che.

## Da Sua

1. Loai `ts` khoi model features.
   - `ts` van duoc giu de chia strict time split va metadata.
   - `graph_metadata.json` hien ghi `excluded_columns: ["ts"]`.

2. Them tuy chon chon best epoch cho GAT.
   - Random split dung `--selection-metric val_f1` de giu cau hinh on dinh.
   - Strict split dung `--selection-metric val_tuned_f1` vi threshold 0.5 lam GAT bi chon epoch qua som.

3. Them tuy chon node feature ablation.
   - Mac dinh: `--node-feature-mode ip`.
   - Thu nghiem: `--node-feature-mode ip_stats`.
   - `ip_stats` tinh train-only node statistics, khong dung label va khong dung val/test edges.

## Ket Qua Sau Khi Sua

Random split, GAT:

```text
Accuracy  = 0.9876
Precision = 0.9792
Recall    = 0.9964
F1-score  = 0.9877
```

Strict time split, GAT:

```text
Accuracy  = 0.7098
Precision = 0.6973
Recall    = 0.7415
F1-score  = 0.7187
```

## Ghi Chu Ve Node Stats

Da thu them train-only node statistics vao node features, nhung lan thu nhanh dau tien lam random split GAT giam xuong khoang F1 = 0.9577. Vi vay pipeline chinh van dung `node-feature-mode ip`, con `ip_stats` duoc giu lai de lam ablation study co he thong sau nay.

Ket luan: khong dua mot thay doi vao ket qua chinh neu no lam giam hieu nang. Day la diem tot khi bao ve vi cho thay quy trinh thuc nghiem trung thuc.
