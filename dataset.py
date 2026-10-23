import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import CLASSES, IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(train=True):
    ops = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]
    return transforms.Compose(ops)

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, file_list=None, train=False):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        if file_list:
            with open(file_list, "r", encoding="utf-8") as f:
                names = {x.strip() for x in f if x.strip()}
            self.df = self.df[self.df["Image Index"].isin(names)].reset_index(drop=True)
        self.transform = build_transform(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        labels = torch.tensor(
            [1.0 if c in str(row["Finding Labels"]).split("|") else 0.0 for c in CLASSES],
            dtype=torch.float32
        )
        return image, labels, row["Image Index"]

def get_class_counts(dataset):
    counts = torch.zeros(len(CLASSES))
    for _, y, _ in dataset:
        counts += y
    return counts.tolist()
