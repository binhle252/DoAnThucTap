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
8. Track improvement decisions:
   - `report/gat_improvement_log.md`

## Commands

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.preprocess_binary --rows-per-class 50000
python -m src.build_graph_arrays --node-feature-mode ip
python -m src.train_gat --epochs 50 --hidden-channels 64 --heads 4 --early-stopping --patience 8 --selection-metric val_f1 --message-passing-edges all
python -m src.train_baselines
python -m src.generate_figures
python -m src.demo_predict
```

Strict split commands:

```powershell
python -m src.create_strict_splits --strategy class_time --rows-per-class 50000 --output-dir data\strict_time_balanced
python -m src.build_graph_arrays --processed-dir data\strict_time_balanced --output-dir data\strict_time_balanced --node-feature-mode ip
python -m src.train_gat --arrays-path data\strict_time_balanced\graph_arrays.npz --model-dir models\strict_time_balanced --results-dir results\strict_time_balanced --epochs 50 --hidden-channels 64 --heads 4 --early-stopping --patience 8 --selection-metric val_tuned_f1 --message-passing-edges train
python -m src.train_baselines --arrays-path data\strict_time_balanced\graph_arrays.npz --model-dir models\strict_time_balanced --results-dir results\strict_time_balanced
```

## Graph Design

- Node: IP address
- Edge: traffic flow from `id.orig_h` to `id.resp_h`
- Edge label: `binary_label`
- Node features: IP-derived features
- Edge features: numeric traffic features and one-hot categorical traffic features
- Anti-leakage policy: `ts` is kept for time-aware splitting and metadata, but it is excluded from model input features.

## Latest Validation-Controlled Result

The improved CPU run on a balanced 100k-flow sample used validation F1 for best-epoch selection and validation F1 for threshold tuning. The test split was only used for final reporting.

- Best epoch: 21
- Selected threshold: 0.5572
- Test Accuracy: 0.9876
- Test Precision: 0.9792
- Test Recall: 0.9964
- Test F1-score: 0.9877
- Test ROC-AUC: 0.9899
- Test PR-AUC: 0.9893

Test confusion matrix:

```text
TN = 7341
FP = 159
FN = 27
TP = 7473
```

The train-test F1 gap is about -0.0004, so the model does not show the usual pattern of severe memorization where train score is much higher than test score.

## Baseline Comparison

Baselines were trained on the same `edge_attr` features and train/validation/test masks as the GAT experiment.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| RandomForest | 0.9998 | 0.9999 | 0.9997 | 0.9998 |
| GAT | 0.9876 | 0.9792 | 0.9964 | 0.9877 |
| MLP | 0.9817 | 0.9999 | 0.9636 | 0.9814 |
| LogisticRegression | 0.9161 | 0.8843 | 0.9576 | 0.9195 |
| DummyMostFrequent | 0.5000 | 0.5000 | 1.0000 | 0.6667 |

The Random Forest score is still almost perfect after removing `ts`, which suggests that the random split is easy and the flow-level features are highly separable.

## Strict Time-Aware Split

The strict split keeps binary classes balanced while preserving temporal order inside each class. GAT is trained with `--message-passing-edges train`, so validation/test edges are not used for graph message passing.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.9985 | 0.9972 | 0.9999 | 0.9985 |
| MLP | 0.8336 | 0.9978 | 0.6687 | 0.8007 |
| GAT | 0.7098 | 0.6973 | 0.7415 | 0.7187 |
| RandomForest | 0.5133 | 0.5067 | 0.9999 | 0.6726 |
| DummyMostFrequent | 0.5000 | 0.5000 | 1.0000 | 0.6667 |

This stricter result is useful for discussion: random split is easy, while stricter time-aware graph evaluation is harder. With validation-tuned epoch selection, GAT improves over Dummy and Random Forest, but it is still below MLP and Logistic Regression.

## Anti-Leakage Notes

- The label column is not used as a feature.
- The raw unique flow id `uid` is dropped.
- The timestamp column `ts` is excluded from model features and used only for time-aware splitting/metadata.
- The model checkpoint is selected by a validation-only metric, not test F1.
- The classification threshold is selected on validation, not test.
- The test split is reported after selection.
- Train-only node statistical features are available for ablation through `--node-feature-mode ip_stats`, but the main reported run uses `--node-feature-mode ip` because the first node-stat trial did not improve the main result.
- For a stricter extension, use an inductive split by time or by unseen IP addresses.
