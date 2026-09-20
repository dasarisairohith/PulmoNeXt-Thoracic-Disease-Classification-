import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset import ChestXrayDataset, get_class_counts
from model import PulmoNeXt
from losses import ClassBalancedFocalLoss
from config import *
from utils import seed_everything, get_device

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            p = torch.sigmoid(model(x.to(device))).cpu()
            ys.append(y)
            ps.append(p)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    aucs = []
    for i in range(NUM_CLASSES):
        if len(set(y[:, i])) > 1:
            aucs.append(roc_auc_score(y[:, i], p[:, i]))
    return float(sum(aucs) / len(aucs))

def main():
    seed_everything(SEED)
    device = get_device()
    print("Device:", device)

    train_ds = ChestXrayDataset(CSV_PATH, IMAGE_DIR, TRAIN_LIST, train=True)
    val_ds = ChestXrayDataset(CSV_PATH, IMAGE_DIR, TEST_LIST, train=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    counts = get_class_counts(train_ds)
    model = PulmoNeXt().to(device)
    criterion = ClassBalancedFocalLoss(counts, BETA, GAMMA).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_auc = -1
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={running/len(train_loader):.4f} | mean ROC-AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "classes": CLASSES}, CHECKPOINT)
            print("Saved:", CHECKPOINT)

if __name__ == "__main__":
    main()
