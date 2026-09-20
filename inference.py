import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import PulmoNeXt
from config import *
from utils import get_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    device = get_device()
    model = PulmoNeXt(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["model"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])

    x = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        p = torch.sigmoid(model(x))[0].cpu()

    for c, score in sorted(zip(CLASSES, p.tolist()), key=lambda z: -z[1]):
        print(f"{c:22s} {score:.4f}")

if __name__ == "__main__":
    main()
