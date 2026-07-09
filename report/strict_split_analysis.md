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
| Logistic Regression | 0.9990 | 0.9981 | 0.9999 | 0.9990 |
| MLP | 0.9957 | 0.9915 | 1.0000 | 0.9958 |
| GAT | 0.7569 | 0.8514 | 0.6224 | 0.7191 |
| Random Forest | 0.5115 | 0.5058 | 0.9999 | 0.6718 |
| Dummy | 0.5000 | 0.5000 | 1.0000 | 0.6667 |

## Nhan Xet

Ket qua strict split cho thay edge features cua IoT-23 rat manh. Logistic Regression va MLP van dat diem cao, nghia la nhieu dau hieu phan biet benign/malicious nam truc tiep trong cac feature cua flow.

GAT trong cau hinh strict dung `message_passing_edges=train`, tuc khong dung val/test edges de tao embedding. Ket qua GAT thap hon, cho thay neu khong cho mo hinh nhin cau truc test graph, viec tong quat hoa sang traffic moi kho hon.

Day la mot ket qua tot ve mat bao ve: no cho thay sinh vien khong chi bao cao mot con so dep, ma con kiem tra mo hinh duoi dieu kien nghiem ngat hon va nhan dien duoc gioi han cua GAT.

## Cach Giai Thich Khi Bao Ve

Co the noi:

```text
O random split, cac mo hinh dat diem rat cao vi train/test co phan phoi kha giong nhau. De tranh nghi ngo hoc thuoc, em thuc hien them strict split theo thoi gian. Ket qua cho thay khi chi message passing tren train graph, GAT giam diem, trong khi Logistic Regression va MLP van rat manh do edge features da co tinh phan biet cao. Dieu nay cho thay bai toan IoT-23 binary detection co nhieu dau hieu ro trong flow-level features, va viec danh gia theo split nghiem ngat la can thiet.
```

## Huong Cai Thien GAT Sau Strict Split

1. Them node feature thong ke tu train graph: degree, in-degree, out-degree, mean bytes, mean packets.
2. Thu GraphSAGE/GATv2 de so sanh voi GATConv.
3. Train voi inductive mini-batch thay vi full-graph.
4. Thu edge classifier manh hon sau node embedding.
5. Lam ablation study: co/khong co edge_attr, message passing all/train, heads 1/2/4.
