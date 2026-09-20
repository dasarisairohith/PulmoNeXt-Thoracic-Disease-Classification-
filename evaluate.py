import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from dataset import ChestXrayDataset
from model import PulmoNeXt
from config import *
from utils import get_device

def main():
    device = get_device()
    ds = ChestXrayDataset(CSV_PATH, IMAGE_DIR, TEST_LIST, train=False)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = PulmoNeXt(pretrained=False).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            ys.append(y.numpy())
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)

    rows = []
    for i, c in enumerate(CLASSES):
        if len(np.unique(y[:, i])) < 2:
            pr = roc = float("nan")
        else:
            pr = average_precision_score(y[:, i], p[:, i])
            roc = roc_auc_score(y[:, i], p[:, i])
        rows.append([c, pr, roc,
                     precision_score(y[:, i], pred[:, i], zero_division=0),
                     recall_score(y[:, i], pred[:, i], zero_division=0),
                     f1_score(y[:, i], pred[:, i], zero_division=0)])

    with open("metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Class","PR-AUC","ROC-AUC","Precision","Recall","F1"])
        w.writerows(rows)

    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
