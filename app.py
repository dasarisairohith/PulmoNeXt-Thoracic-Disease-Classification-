import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from config import CHECKPOINT, CLASSES
from dataset import build_transform
from utils import get_device

CHECKPOINT_PATH = Path(__file__).resolve().parent / CHECKPOINT

FINDING_CONTEXT = {
    "Atelectasis": "Radiologists assess for lung volume loss or partial collapse.",
    "Cardiomegaly": "Radiologists assess the size of the heart silhouette, accounting for image projection.",
    "Consolidation": "Radiologists assess for air-space opacity that may obscure normal lung markings.",
    "Edema": "Radiologists assess for patterns such as interstitial or air-space opacity and vascular congestion.",
    "Effusion": "Radiologists assess for pleural fluid, including blunting of the costophrenic angles.",
    "Emphysema": "Radiologists may assess for hyperinflation and flattened diaphragms.",
    "Fibrosis": "Radiologists assess for persistent reticular or other interstitial markings.",
    "Hernia": "Radiologists assess for abdominal contents or a hernia projecting above the diaphragm.",
    "Infiltration": "This is a broad, nonspecific opacity label; radiologists assess the location and pattern of increased lung opacity.",
    "Mass": "Radiologists assess for a focal mass-like opacity and its size, shape, and location.",
    "Nodule": "Radiologists assess for a small, focal rounded opacity and its size, shape, and location.",
    "Pleural Thickening": "Radiologists assess for thickening or scarring along the pleural surface.",
    "Pneumonia": "Radiologists assess for air-space opacity in a clinical context; this can overlap with other causes of opacity.",
    "Pneumothorax": "Radiologists assess for a pleural line with reduced lung markings beyond it.",
}


@st.cache_resource
def load_model(checkpoint_path: Path):
    device = get_device()
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("Checkpoint must contain a 'model' state dictionary.")

        from model import PulmoNeXt

        model = PulmoNeXt(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        model_type = "pulmonext"
        model_labels = CLASSES
    else:
        package_spec = importlib.util.find_spec("torchxrayvision")
        if package_spec is None or package_spec.origin is None:
            raise ImportError("Install the torchxrayvision package to use pretrained weights.")

        # TorchXRayVision imports its bundled baseline package as top-level `model`.
        baseline_path = (
            Path(package_spec.origin).parent
            / "baseline_models"
            / "jfhealthcare"
        )
        sys.path.insert(0, str(baseline_path))
        try:
            import torchxrayvision as xrv
        finally:
            sys.path.remove(str(baseline_path))

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
        model_type = "torchxrayvision"
        model_labels = model.pathologies

    model.eval()
    return model, device, model_type, model_labels


def prepare_input(image, model_type, device):
    if model_type == "pulmonext":
        tensor = build_transform(train=False)(image)
    else:
        import torchxrayvision as xrv
        from torchvision import transforms

        grayscale = np.asarray(image.convert("L"), dtype=np.float32)
        normalized = xrv.datasets.normalize(grayscale, 255)[None, ...]
        transform = transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224),
            ]
        )
        tensor = torch.from_numpy(transform(normalized))
    return tensor.unsqueeze(0).to(device)


def make_gradcam(model, image_tensor, class_index, image, model_type):
    activations = []
    gradients = []
    if model_type == "pulmonext":
        target_layer = model.backbone.features[-1]
    else:
        target_layer = model.features.denseblock4

    def capture_activation(_, __, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient.detach()))

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        score = model(image_tensor)[0, class_index]
        score.backward()
    finally:
        hook.remove()

    activation = activations[0][0].detach().cpu().numpy()
    gradient = gradients[0][0].cpu().numpy()
    weights = gradient.mean(axis=(1, 2))
    heatmap = np.maximum((weights[:, None, None] * activation).sum(axis=0), 0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8

    colors = plt.get_cmap("jet")(heatmap)[..., :3]
    heatmap_image = Image.fromarray(np.uint8(colors * 255)).resize(image.size)
    return heatmap_image


def describe_model_result(class_name, scores_by_class):
    ranked_classes = sorted(
        scores_by_class,
        key=scores_by_class.get,
        reverse=True,
    )
    rank = ranked_classes.index(class_name) + 1
    score = scores_by_class[class_name]
    return (
        f"The model ranked **{class_name}** #{rank} of {len(ranked_classes)} "
        f"for this image (score **{score:.1%}**). "
        f"{FINDING_CONTEXT[class_name]} "
        "This is general context about what clinicians look for; the model "
        "does not confirm that this finding is present in the uploaded image."
    )


def main():
    st.set_page_config(page_title="PulmoNeXt X-ray Analyzer", layout="wide")
    st.title("PulmoNeXt X-ray Analyzer")
    st.write(
        "Upload a chest X-ray to view the model's 14 disease probabilities "
        "and a Grad-CAM explanation."
    )
    st.warning(
        "Research demonstration only — results are not medical diagnoses "
        "and must not guide clinical decisions."
    )
    st.caption(
        "If no locally trained PulmoNeXt checkpoint is present, the app uses "
        "publicly pretrained TorchXRayVision weights (downloaded on first use)."
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is None:
        st.info("Upload a PNG or JPEG image to get started.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"Could not read the uploaded image: {error}")
        return

    try:
        model, device, model_type, model_labels = load_model(CHECKPOINT_PATH)
    except (
        ImportError,
        KeyError,
        requests.exceptions.RequestException,
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        st.error(f"Could not load an X-ray model: {error}")
        st.info(
            "Check your internet connection for the initial pretrained-weight "
            "download, or add a trained pulmoNeXt_best.pth checkpoint."
        )
        return

    image_tensor = prepare_input(image, model_type, device)
    with torch.no_grad():
        outputs = model(image_tensor)[0].cpu().numpy()

    label_indices = {
        label.replace("_", " "): index
        for index, label in enumerate(model_labels)
    }
    scores_by_class = {}
    for class_name in CLASSES:
        model_label = (
            "Pleural_Thickening"
            if class_name == "Pleural Thickening"
            and model_type == "torchxrayvision"
            else class_name
        )
        normalized_label = model_label.replace("_", " ")
        if normalized_label not in label_indices:
            st.error(f"The loaded model does not provide a {class_name} output.")
            return
        score = outputs[label_indices[normalized_label]]
        if model_type == "pulmonext":
            score = 1 / (1 + np.exp(-score))
        scores_by_class[class_name] = float(score)

    ranked = sorted(scores_by_class.items(), key=lambda item: item[1], reverse=True)
    class_names, scores = zip(*ranked)

    image_column, chart_column = st.columns([1, 1])
    with image_column:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)
    with chart_column:
        st.subheader("Predicted disease probabilities")
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.barh(class_names[::-1], scores[::-1], color="#3182bd")
        axis.set_xlim(0, 1)
        axis.set_xlabel("Probability")
        axis.set_ylabel("Disease class")
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)

    selected_class = st.selectbox(
        "Grad-CAM class",
        class_names,
        index=0,
    )
    model_label = (
        "Pleural_Thickening"
        if selected_class == "Pleural Thickening"
        and model_type == "torchxrayvision"
        else selected_class
    )
    class_index = model_labels.index(model_label)
    try:
        heatmap = make_gradcam(
            model,
            image_tensor,
            class_index,
            image,
            model_type,
        )
    except (RuntimeError, IndexError) as error:
        st.error(f"Could not generate the Grad-CAM visualization: {error}")
        return

    overlay = Image.blend(image, heatmap.convert("RGB"), alpha=0.4)
    st.subheader(f"Grad-CAM: {selected_class}")
    st.image(
        overlay,
        caption=(
            "Highlighted regions indicate image areas that influenced this "
            "model output; they do not prove a disease or identify a specific sign."
        ),
        use_container_width=True,
    )
    st.subheader(f"Why did the model flag {selected_class}?")
    st.markdown(describe_model_result(selected_class, scores_by_class))
    st.caption(
        "The score is a model output, not the chance that a patient has the disease. "
        "Only a qualified clinician can interpret a chest X-ray with the patient's "
        "history and other tests."
    )


if __name__ == "__main__":
    main()
