# IDS GAT IoT-23

Project topic: malicious traffic detection in computer networks using Graph Attention Network (GAT) with the IoT-23 dataset.

## Current Pipeline

1. Explore the raw IoT-23 CSV:
   - `notebooks/01_Data_Exploration.ipynb`
2. Create a balanced binary dataset:
   - `notebooks/02_Preprocessing.ipynb`
   - `src/preprocess_binary.py`
3. Build graph-ready arrays:
   - `notebooks/03_Graph_Construction.ipynb`
   - `src/build_graph_arrays.py`
4. Train and evaluate GAT:
   - `notebooks/04_GAT_Training.ipynb`
   - `src/train_gat.py`
5. Demo prediction for one new traffic flow:
   - `notebooks/05_Demo_Prediction.ipynb`
   - `src/demo_predict.py`
6. Compare GAT with graphless baselines:
   - `notebooks/06_Baseline_Comparison.ipynb`
   - `src/train_baselines.py`
7. Evaluate with stricter time-aware split:
   - `notebooks/07_Strict_Time_Split.ipynb`
   - `src/create_strict_splits.py`

## Commands

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.preprocess_binary --rows-per-class 50000
python -m src.build_graph_arrays
python -m src.train_gat --epochs 50 --hidden-channels 64 --heads 4 --early-stopping --patience 8
python -m src.train_baselines
python -m src.generate_figures
python -m src.demo_predict
```

Strict split commands:

```powershell
python -m src.create_strict_splits --strategy class_time --rows-per-class 50000 --output-dir data\strict_time_balanced
python -m src.build_graph_arrays --processed-dir data\strict_time_balanced --output-dir data\strict_time_balanced
python -m src.train_gat --arrays-path data\strict_time_balanced\graph_arrays.npz --model-dir models\strict_time_balanced --results-dir results\strict_time_balanced --epochs 50 --hidden-channels 64 --heads 4 --early-stopping --patience 8 --message-passing-edges train
python -m src.train_baselines --arrays-path data\strict_time_balanced\graph_arrays.npz --model-dir models\strict_time_balanced --results-dir results\strict_time_balanced
```

## Graph Design

- Node: IP address
- Edge: traffic flow from `id.orig_h` to `id.resp_h`
- Edge label: `binary_label`
- Node features: IP-derived features
- Edge features: numeric traffic features and one-hot categorical traffic features

## Latest Validation-Controlled Result

The improved CPU run on a balanced 100k-flow sample used validation F1 for best-epoch selection and validation F1 for threshold tuning. The test split was only used for final reporting.

- Best epoch: 39
- Selected threshold: 0.6716
- Test Accuracy: 0.9889
- Test Precision: 0.9793
- Test Recall: 0.9989
- Test F1-score: 0.9890
- Test ROC-AUC: 0.9990
- Test PR-AUC: 0.9989

Test confusion matrix:

```text
TN = 7342
FP = 158
FN = 8
TP = 7492
```

The train-test F1 gap is about -0.0005, so the model does not show the usual pattern of severe memorization where train score is much higher than test score.

## Baseline Comparison

Baselines were trained on the same `edge_attr` features and train/validation/test masks as the GAT experiment.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| RandomForest | 0.9999 | 0.9999 | 1.0000 | 0.9999 |
| MLP | 0.9943 | 0.9997 | 0.9888 | 0.9942 |
| GAT | 0.9889 | 0.9793 | 0.9989 | 0.9890 |
| LogisticRegression | 0.9542 | 0.9799 | 0.9275 | 0.9529 |
| DummyMostFrequent | 0.5000 | 0.5000 | 1.0000 | 0.6667 |

The Random Forest score is almost perfect on the current random split, which suggests that this split is relatively easy. The next essential extension is a stricter evaluation split, such as time-based split or unseen-IP split.

## Strict Time-Aware Split

The strict split keeps binary classes balanced while preserving temporal order inside each class. GAT is trained with `--message-passing-edges train`, so validation/test edges are not used for graph message passing.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.9990 | 0.9981 | 0.9999 | 0.9990 |
| MLP | 0.9957 | 0.9915 | 1.0000 | 0.9958 |
| GAT | 0.7569 | 0.8514 | 0.6224 | 0.7191 |
| RandomForest | 0.5115 | 0.5058 | 0.9999 | 0.6718 |
| DummyMostFrequent | 0.5000 | 0.5000 | 1.0000 | 0.6667 |

This stricter result is useful for discussion: random split is easy, while stricter time-aware graph evaluation reveals harder generalization behavior for GAT.

## Anti-Leakage Notes

- The label column is not used as a feature.
- The raw unique flow id `uid` is dropped.
- The model checkpoint is selected by validation F1, not test F1.
- The classification threshold is selected on validation, not test.
- The test split is reported after selection.
- For a stricter extension, use an inductive split by time or by unseen IP addresses.
