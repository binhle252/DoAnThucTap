# Quy Trinh Danh Gia De Tranh Nghi Ngo Hoc Thuoc

## Muc Tieu

Muc tieu cua mo hinh khong phai la dat 100% tren test set bang moi gia, vi ket qua hoan hao tuyet doi co the gay nghi ngo ve viec hoc thuoc hoac ro ri du lieu. Muc tieu hop ly hon la dat cac chi so tiem can muc hoan hao, dong thoi chung minh duoc quy trinh danh gia la nghiem tuc.

## Cach Danh Gia

- Chia du lieu thanh train, validation va test.
- Train set dung de cap nhat trong so mo hinh.
- Validation set dung de chon epoch tot nhat va nguong phan loai.
- Test set chi dung mot lan de bao cao ket qua cuoi cung.
- Khong dung nhan `label` lam dac trung dau vao.
- Loai bo `uid` vi day la ma dinh danh duy nhat cua tung flow.

## Ket Qua Hien Tai

Mo hinh GAT dat ket qua tren test set:

```text
Accuracy  = 0.9889
Precision = 0.9793
Recall    = 0.9989
F1-score  = 0.9890
ROC-AUC   = 0.9990
PR-AUC    = 0.9989
```

Confusion matrix tren test set:

```text
TN = 7342
FP = 158
FN = 8
TP = 7492
```

## Giai Thich Khi Bao Ve

Ket qua cao vi bai toan hien tai la binary classification: `Benign` so voi `Malicious`. Cac hanh vi tan cong trong IoT-23 co nhieu dau hieu manh trong cac feature nhu protocol, connection state, packet count, byte count va dia chi IP. Tuy nhien, ket qua khong phai 100%, van co false positive va false negative, nen khong co dau hieu hoan hao bat thuong.

Train-test gap cua F1-score rat nho, khoang -0.0005. Dieu nay cho thay mo hinh khong co hien tuong train score qua cao nhung test score thap, la dau hieu thuong gap khi hoc thuoc.

## Huong Mo Rong Nghiem Ngat Hon

De danh gia chat che hon, co the mo rong theo cac huong:

1. Chia du lieu theo thoi gian: train tren traffic cu, test tren traffic moi.
2. Chia theo IP: test tren cac IP chua xuat hien trong train.
3. Chuyen tu binary classification sang multi-class classification.
4. Chay nhieu random seed va bao cao mean plus standard deviation.
5. So sanh GAT voi baseline nhu Logistic Regression, Random Forest va MLP.
