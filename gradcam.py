import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import PulmoNeXt
from config import *
from utils import get_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--class_name", default="Hernia")
    ap.add_argument("--output", default="gradcam_hernia.png")
    args = ap.parse_args()

    device = get_device()
    model = PulmoNeXt(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["model"])
    model.eval()

    # Target the final ConvNeXt feature block.
    target_layer = model.backbone.features[-1]
    activations, gradients = [], []

    def fwd(_, __, out):
        activations.append(out.detach())

    def bwd(_, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd)
    h2 = target_layer.register_full_backward_hook(bwd)

    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])
    raw = Image.open(args.image).convert("RGB")
    x = tfm(raw).unsqueeze(0).to(device)

    idx = CLASSES.index(args.class_name)
    model.zero_grad()
    score = model(x)[0, idx]
    score.backward()

    a = activations[0][0].cpu().numpy()
    g = gradients[0][0].cpu().numpy()
    weights = g.mean(axis=(1,2))
    cam = np.maximum((weights[:,None,None] * a).sum(axis=0), 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(raw.size)
    plt.figure(figsize=(8,8))
    plt.imshow(raw, cmap="gray")
    plt.imshow(cam_img, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", args.output)

    h1.remove(); h2.remove()

if __name__ == "__main__":
    main()
