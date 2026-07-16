import pandas as pd; \
t=pd.read_csv('data/strict_time_balanced/train.csv'); \
v=pd.read_csv('data/strict_time_balanced/val.csv'); \
te=pd.read_csv('data/strict_time_balanced/test.csv'); \
print('TRAIN');print(t['label'].value_counts()); \
print('VAL');print(v['label'].value_counts()); \
print('TEST');print(te['label'].value_counts())