# Ke Hoach 5 Viec Mo Rong Thiet Yeu

## Muc Tieu 4 Tuan

Hoan thien do an theo huong bam sat de tai IDS + IoT-23 + GAT, co so sanh, co kiem chung, co demo va co giai thich duoc khi bao ve.

## 1. Baseline Comparison

Trang thai: da lam.

Muc dich:

- Tra loi cau hoi: vi sao dung GAT?
- So sanh GAT voi cac mo hinh khong dung graph.

Artifact:

- `src/train_baselines.py`
- `notebooks/06_Baseline_Comparison.ipynb`
- `results/baseline_metrics.json`
- `results/model_comparison.csv`
- `results/figures/model_comparison_f1.png`

Nhan xet hien tai:

- Random Forest dat gan nhu hoan hao tren random split.
- Dieu nay cho thay random split co the qua de.
- Vi vay viec tiep theo rat quan trong la strict split.

## 2. GAT Ablation Study

Trang thai: chua lam.

Can thu:

- GAT co edge features va khong co edge features.
- GAT 1 layer va 2 layer.
- Heads = 1, 2, 4.
- Hidden channels = 32, 64, 128.

Ket qua mong muon:

- Chung minh cau hinh GAT hien tai la co ly do.
- Biet thanh phan nao giup tang F1-score.

## 3. Strict Split

Trang thai: da lam ban dau.

Can lam mot trong hai cach:

- Time-based split: train tren traffic cu, test tren traffic moi.
- Unseen-IP split: test tren IP chua xuat hien trong train.

Da co:

- `src/create_strict_splits.py`
- `notebooks/07_Strict_Time_Split.ipynb`
- `data/strict_time_balanced/`
- `results/strict_time_balanced/`
- `report/strict_split_analysis.md`

Ket qua strict split cho thay GAT giam diem khi chi message passing tren train graph, trong khi Logistic Regression va MLP van rat manh. Day la bang chung huu ich de thao luan ve gioi han cua random split va vai tro cua edge features.

## 4. Visualization

Trang thai: da lam mot phan.

Da co:

- `results/figures/model_comparison_f1.png`
- `results/figures/gat_training_curve.png`
- `results/figures/gat_confusion_matrix_test.png`

Can them:

- Label distribution.
- Dataset pipeline diagram.
- Graph construction diagram.

## 5. Demo Tot Hon

Trang thai: co CLI demo, chua co giao dien.

Can mo rong:

- Nhap mot traffic flow moi.
- Hien nhan du doan.
- Hien xac suat Benign/Malicious.
- Hien threshold dang dung.
- Co the dung Streamlit neu can giao dien truc quan.

## Thu Tu Lam Tiep

1. Strict split.
2. GAT ablation study.
3. Hoan thien visualization.
4. Demo giao dien.
5. Viet bao cao va slide.
