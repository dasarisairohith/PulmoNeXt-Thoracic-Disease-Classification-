import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE,
    BETA,
    CHECKPOINT,
    CLASSES,
    CSV_PATH,
    EPOCHS,
    GAMMA,
    IMAGE_DIR,
    LR,
    NUM_CLASSES,
    NUM_WORKERS,
    SEED,
    TRAIN_LIST,
    VALIDATION_FRACTION,
    WEIGHT_DECAY,
)
from dataset import ChestXrayDataset
from losses import ClassBalancedFocalLoss
from model import PulmoNeXt
from utils import get_device, seed_everything


def evaluate(model, loader, device):
    model.eval()
    all_targets = []
    all_scores = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
            all_targets.append(targets.numpy())
            all_scores.append(torch.sigmoid(logits).float().cpu().numpy())

    targets = np.concatenate(all_targets)
    scores = np.concatenate(all_scores)
    average_precisions = []
    roc_aucs = []
    for class_index in range(NUM_CLASSES):
        class_targets = targets[:, class_index]
        if np.unique(class_targets).size < 2:
            continue
        average_precisions.append(
            average_precision_score(class_targets, scores[:, class_index])
        )
        roc_aucs.append(roc_auc_score(class_targets, scores[:, class_index]))

    if not average_precisions:
        raise ValueError(
            "Validation split has no disease class with positive and negative "
            "examples; cannot select a model."
        )
    return float(np.mean(average_precisions)), float(np.mean(roc_aucs))


def split_training_data():
    full_training_dataset = ChestXrayDataset(
        CSV_PATH,
        IMAGE_DIR,
        TRAIN_LIST,
        train=False,
    )
    if "Patient ID" not in full_training_dataset.df:
        raise ValueError(
            "ChestX-ray14 metadata must include 'Patient ID' for patient-wise "
            "train/validation separation."
        )
    if len(full_training_dataset) < 2:
        raise ValueError("The official training list has too few images to split.")

    patient_ids = full_training_dataset.df["Patient ID"].to_numpy()
    if full_training_dataset.df["Patient ID"].isna().any():
        raise ValueError("Patient IDs must be present for every training image.")
    if np.unique(patient_ids).size < 2:
        raise ValueError("At least two distinct patients are required for splitting.")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(patient_ids)), groups=patient_ids)
    )

    train_patients = set(patient_ids[train_indices])
    validation_patients = set(patient_ids[validation_indices])
    if train_patients & validation_patients:
        raise RuntimeError("Patient leakage detected in the validation split.")

    augmented_dataset = ChestXrayDataset(
        CSV_PATH,
        IMAGE_DIR,
        TRAIN_LIST,
        train=True,
    )
    return (
        Subset(augmented_dataset, train_indices.tolist()),
        Subset(full_training_dataset, validation_indices.tolist()),
        full_training_dataset.df.iloc[train_indices]["Finding Labels"].tolist(),
    )


def count_positive_labels(label_rows):
    positive_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_indices = {class_name: index for index, class_name in enumerate(CLASSES)}
    for label_row in label_rows:
        for label in str(label_row).split("|"):
            class_index = class_indices.get(label)
            if class_index is not None:
                positive_counts[class_index] += 1
    return positive_counts.tolist()


def main():
    if not TRAIN_LIST:
        raise ValueError("Configure the ChestX-ray14 training list.")

    seed_everything(SEED)
    device = get_device()
    print("Device:", device)
    print("Training split: official train_val_list, split by patient.")
    print("The official test_list is held out for evaluate.py.")

    train_dataset, validation_dataset, train_labels = split_training_data()
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    counts = count_positive_labels(train_labels)
    model = PulmoNeXt().to(device)
    criterion = ClassBalancedFocalLoss(counts, BETA, GAMMA).to(device)

    head_parameters = list(model.backbone.classifier[2].parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": LR * 0.1},
            {"params": head_parameters, "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_validation_ap = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        validation_ap, validation_auc = evaluate(
            model,
            validation_loader,
            device,
        )
        scheduler.step(validation_ap)
        mean_loss = running_loss / max(len(train_loader), 1)
        backbone_lr, head_lr = (
            parameter_group["lr"] for parameter_group in optimizer.param_groups
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={mean_loss:.4f} | "
            f"validation macro PR-AUC={validation_ap:.4f} | "
            f"validation macro ROC-AUC={validation_auc:.4f} | "
            f"backbone lr={backbone_lr:.2e} | head lr={head_lr:.2e}"
        )

        if validation_ap > best_validation_ap:
            best_validation_ap = validation_ap
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")

    torch.save(
        {
            "model": best_state,
            "classes": CLASSES,
            "best_epoch": best_epoch,
            "validation_macro_pr_auc": best_validation_ap,
            "architecture": "ConvNeXt-Tiny",
            "validation_split": "patient-wise from train_val_list",
        },
        CHECKPOINT,
    )
    print(
        f"Saved best validation checkpoint: {CHECKPOINT} "
        f"(epoch={best_epoch}, macro PR-AUC={best_validation_ap:.4f})"
    )


if __name__ == "__main__":
    main()
