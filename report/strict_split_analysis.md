# Danh Gia Strict Split Theo Thoi Gian

## Ly Do Lam Strict Split

Ket qua random split truoc do rat cao. Random Forest gan nhu dat diem tuyet doi, cho thay random split co the qua de vi cac mau train/test co phan phoi rat giong nhau. De tranh nghi ngo mo hinh hoc thuoc, can them mot cach chia du lieu nghiem ngat hon.

## Cach Chia

Phien ban hien tai dung `class_time split`:

- Trong tung lop `Benign` va `Malicious`, sap xep du lieu theo timestamp `ts`.
- 70% cu hon dung cho train.
- 15% tiep theo dung cho validation.
- 15% moi hon dung cho test.
- Moi split van can bang Benign/Malicious de cac chi so Accuracy, Precision, Recall va F1-score co y nghia.

So dong:

```text
Train: 70,000
Val  : 15,000
Test : 15,000
```

## Ket Qua Strict Split

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9985 | 0.9972 | 0.9999 | 0.9985 |
| MLP | 0.8336 | 0.9978 | 0.6687 | 0.8007 |
| GAT | 0.7098 | 0.6973 | 0.7415 | 0.7187 |
| Random Forest | 0.5133 | 0.5067 | 0.9999 | 0.6726 |
| Dummy | 0.5000 | 0.5000 | 1.0000 | 0.6667 |

## Nhan Xet

Ket qua strict split sau khi loai `ts` cho thay edge features cua IoT-23 van rat manh doi voi mot so baseline. Logistic Regression van dat diem cao, nghia la nhieu dau hieu phan biet benign/malicious nam truc tiep trong cac feature cua flow va categorical pattern.

GAT trong cau hinh strict dung `message_passing_edges=train`, tuc khong dung val/test edges de tao embedding. Khi chon best epoch bang `val_tuned_f1`, GAT tang len F1-score khoang 0.7187, cao hon Dummy va Random Forest trong strict split. Tuy nhien, GAT van thap hon MLP va Logistic Regression, cho thay cau hinh GAT hien tai can cai tien them de canh tranh voi cac baseline flow-level.

Day la mot ket qua tot ve mat bao ve: no cho thay sinh vien khong chi bao cao mot con so dep, ma con kiem tra mo hinh duoi dieu kien nghiem ngat hon, loai timestamp shortcut, cai thien cach chon epoch cua GAT va nhan dien duoc gioi han cua mo hinh.

## Cach Giai Thich Khi Bao Ve

Co the noi:

```text
O random split, cac mo hinh dat diem cao vi train/test co phan phoi kha giong nhau. De tranh nghi ngo hoc thuoc, em thuc hien them strict split theo thoi gian va loai `ts` khoi feature model. Ket qua cho thay khi chi message passing tren train graph, GAT kho hon nhieu so voi random split nhung van cao hon Dummy va Random Forest sau khi chon epoch bang validation-tuned F1. Logistic Regression van rat manh do flow-level features co tinh phan biet cao. Dieu nay cho thay viec danh gia theo split nghiem ngat la can thiet va GAT can duoc cai tien them.
```

## Huong Cai Thien GAT Sau Strict Split

1. Lam ablation study co he thong: `node-feature-mode ip` so voi `ip_stats`, heads 1/2/4, hidden channels 32/64/128.
2. Thu GraphSAGE/GATv2 de so sanh voi GATConv.
3. Train voi inductive mini-batch thay vi full-graph.
4. Thu edge classifier manh hon sau node embedding.
5. Them unseen-IP split de danh gia traffic tu IP moi.
