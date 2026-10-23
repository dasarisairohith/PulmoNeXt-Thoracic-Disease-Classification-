import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError

from config import CHECKPOINT, CLASSES, IMAGE_SIZE
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


def _preview_image(values, minimum, maximum):
    pixels = np.clip((values - minimum) * (255.0 / (maximum - minimum)), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def _false_color_image(values, minimum, maximum):
    scaled = np.clip((values - minimum) / (maximum - minimum), 0, 1)
    colors = plt.get_cmap("turbo")(scaled)[..., :3]
    return Image.fromarray(np.uint8(colors * 255))


def build_dip_concept_stages(image):
    demo_input = image.convert("L")
    demo_input.thumbnail((512, 512), Image.Resampling.LANCZOS)

    noisy_pixels = np.asarray(demo_input, dtype=np.uint8).copy()
    noise_rng = np.random.default_rng(42)
    noise_mask = noise_rng.random(noisy_pixels.shape) < 0.04
    salt_mask = noise_rng.random(noisy_pixels.shape) < 0.5
    noisy_pixels[noise_mask & salt_mask] = 255
    noisy_pixels[noise_mask & ~salt_mask] = 0
    noisy_image = Image.fromarray(noisy_pixels)
    denoised = noisy_image.filter(ImageFilter.MedianFilter(size=3))
    contrast_stretched = ImageOps.autocontrast(denoised, cutoff=1)
    equalized = ImageOps.equalize(contrast_stretched)

    histogram = np.histogram(
        np.asarray(equalized, dtype=np.uint8),
        bins=256,
        range=(0, 256),
    )[0]
    pixel_count = histogram.sum()
    total_intensity = np.dot(np.arange(256), histogram)
    background_count = 0
    background_sum = 0.0
    between_class_variance = np.zeros(256, dtype=np.float64)
    for threshold in range(256):
        background_count += histogram[threshold]
        if background_count == 0:
            continue
        foreground_count = pixel_count - background_count
        if foreground_count == 0:
            break
        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_count
        foreground_mean = (total_intensity - background_sum) / foreground_count
        between_class_variance[threshold] = (
            background_count
            * foreground_count
            * (background_mean - foreground_mean) ** 2
        )
    threshold = int(np.argmax(between_class_variance))
    segmented = equalized.point(lambda value: 255 if value > threshold else 0)
    opening_input_pixels = np.asarray(segmented, dtype=np.uint8).copy()
    speck_rng = np.random.default_rng(7)
    speck_mask = (
        (speck_rng.random(opening_input_pixels.shape) < 0.005)
        & (opening_input_pixels == 0)
    )
    opening_input_pixels[speck_mask] = 255
    opening_input = Image.fromarray(opening_input_pixels)
    opened = opening_input.filter(ImageFilter.MinFilter(size=3)).filter(
        ImageFilter.MaxFilter(size=3)
    )
    edges = ImageOps.autocontrast(
        equalized.filter(ImageFilter.FIND_EDGES)
    )

    return [
        (
            "Grayscale conversion",
            demo_input,
            "Convert the image to one intensity channel.",
            demo_input,
        ),
        (
            "Synthetic salt-and-pepper noise",
            noisy_image,
            "For demonstration only, add reproducible black and white speckles to 4% of pixels so noise removal is visible. The uploaded image is not changed.",
            demo_input,
        ),
        (
            "Noise reduction",
            denoised,
            "Apply a 3 × 3 median filter to remove the synthetic speckles while preserving larger structures.",
            noisy_image,
        ),
        (
            "Contrast stretching",
            contrast_stretched,
            "Expand the useful intensity range using 1% tail clipping.",
            denoised,
        ),
        (
            "Histogram equalization",
            equalized,
            "Redistribute intensity values to make contrast changes visible.",
            contrast_stretched,
        ),
        (
            "Threshold segmentation",
            segmented,
            f"Separate dark and bright regions with Otsu's automatically selected threshold ({threshold}).",
            equalized,
        ),
        (
            "Morphology input with specks",
            opening_input,
            "For demonstration only, add isolated white pixels to the binary mask so the opening operation's cleanup effect is visible.",
            segmented,
        ),
        (
            "Morphological opening",
            opened,
            "Erode, then dilate, the binary mask using a 3 × 3 square structuring element; isolated specks are removed while larger regions remain.",
            opening_input,
        ),
        (
            "Edge detection",
            edges,
            "Highlight intensity boundaries using a gradient-based edge filter on the enhanced grayscale image.",
            equalized,
        ),
    ]


def prepare_input(image, model_type, device):
    stages = [("Input image", image, "Uploaded chest X-ray.", None)]

    if model_type == "pulmonext":
        tensor = build_transform(train=False)(image)
        from torchvision import transforms

        resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        to_grayscale = transforms.Grayscale(num_output_channels=3)
        resized = resize(image)
        grayscale_rgb = to_grayscale(resized)
        pixel_tensor = transforms.ToTensor()(grayscale_rgb)
        normalized_values = tensor.mean(dim=0).numpy()
        stages.extend(
            [
                (
                    "Resize",
                    resized,
                    f"Resize to {IMAGE_SIZE} × {IMAGE_SIZE} using the training transform.",
                    image,
                ),
                (
                    "Grayscale",
                    grayscale_rgb,
                    "Convert to grayscale and replicate into the model's three channels. Little visible change is expected for an X-ray.",
                    resized,
                ),
                (
                    "Tensor conversion",
                    Image.fromarray(pixel_tensor.permute(1, 2, 0).mul(255).byte().numpy()),
                    "Convert pixels to a 3 × 224 × 224 tensor in [0, 1]; the displayed image looks unchanged because this step changes data format.",
                    grayscale_rgb,
                ),
                (
                    "ImageNet normalization",
                    _false_color_image(normalized_values, -3, 3),
                    "False-color view of channel-averaged normalized values, clipped to [-3, 3]. The model receives three-channel normalized values, not these colors.",
                    Image.fromarray(pixel_tensor.permute(1, 2, 0).mul(255).byte().numpy()),
                ),
            ]
        )
    else:
        import torchxrayvision as xrv

        grayscale = np.asarray(image.convert("L"), dtype=np.float32)
        normalized = xrv.datasets.normalize(grayscale, 255)[None, ...]
        cropped = xrv.datasets.XRayCenterCrop()(normalized)
        resized = xrv.datasets.XRayResizer(IMAGE_SIZE)(cropped)
        tensor = torch.from_numpy(resized)
        stages.extend(
            [
                (
                    "Grayscale",
                    Image.fromarray(grayscale.astype(np.uint8)),
                    "Convert RGB pixels to one intensity channel. Little visible change is expected for an X-ray that is already grayscale.",
                    image,
                ),
                (
                    "Intensity normalization",
                    _false_color_image(normalized[0], -1024, 1024),
                    "False-color view of values scaled to [-1024, 1024]. The model still receives single-channel numeric values, not these colors.",
                    Image.fromarray(grayscale.astype(np.uint8)),
                ),
                (
                    "Center crop",
                    _preview_image(cropped[0], -1024, 1024),
                    "Remove the side margins to make a centered square. The crop area is outlined in the before image.",
                    None,
                ),
                (
                    "Resize",
                    _preview_image(resized[0], -1024, 1024),
                    f"Resample to {IMAGE_SIZE} × {IMAGE_SIZE}; the model receives this lower-resolution tensor.",
                    _preview_image(cropped[0], -1024, 1024),
                ),
            ]
        )
        crop_size = min(grayscale.shape)
        left = grayscale.shape[1] // 2 - crop_size // 2
        top = grayscale.shape[0] // 2 - crop_size // 2
        crop_outline = Image.fromarray(grayscale.astype(np.uint8)).convert("RGB")
        drawing = ImageDraw.Draw(crop_outline)
        drawing.rectangle(
            (left, top, left + crop_size - 1, top + crop_size - 1),
            outline=(255, 40, 40),
            width=max(2, min(grayscale.shape) // 250),
        )
        stages[3] = (
            stages[3][0],
            stages[3][1],
            stages[3][2],
            crop_outline,
        )
    return tensor.unsqueeze(0).to(device), stages


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

    image_tensor, dip_stages = prepare_input(image, model_type, device)
    st.subheader("Image processing workflow")
    st.caption(
        "Follow each stage in order. The final stage is the tensor passed to the model."
    )
    total_steps = len(dip_stages)
    for step_number, (title, stage_image, description, before_image) in enumerate(
        dip_stages,
        start=1,
    ):
        with st.container(border=True):
            st.markdown(f"**Step {step_number} of {total_steps}: {title}**")
            st.caption(f"{description} · Stage size: {stage_image.width} × {stage_image.height}px")
            if before_image is None:
                preview = stage_image.copy()
                preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
                st.image(
                    preview,
                    width=512,
                    alt=f"Step {step_number}: {title} chest X-ray processing stage",
                )
            else:
                before_column, after_column = st.columns(2)
                for column, label, source_image in (
                    (before_column, "Before", before_image),
                    (after_column, "After", stage_image),
                ):
                    with column:
                        st.markdown(f"_{label}_")
                        preview = source_image.copy()
                        if title == "Resize" and label == "After":
                            preview = preview.resize(
                                (IMAGE_SIZE * 3, IMAGE_SIZE * 3),
                                Image.Resampling.NEAREST,
                            )
                        else:
                            preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        st.image(
                            preview,
                            width="stretch",
                            alt=f"Step {step_number} {label.lower()} view for {title}",
                        )

    st.subheader("Classic DIP concepts")
    st.caption(
        "An educational image-processing sequence: grayscale → denoising → "
        "contrast enhancement → segmentation → morphology → edges."
    )
    st.info(
        "These demonstration operations are separate from the model pipeline; "
        "they do not change the image used for prediction."
    )
    concept_stages = build_dip_concept_stages(image)
    for step_number, (title, stage_image, description, before_image) in enumerate(
        concept_stages,
        start=1,
    ):
        with st.container(border=True):
            st.markdown(
                f"**DIP step {step_number} of {len(concept_stages)}: {title}**"
            )
            st.caption(description)
            before_column, after_column = st.columns(2)
            for column, label, source_image in (
                (before_column, "Before", before_image),
                (after_column, "After", stage_image),
            ):
                with column:
                    st.markdown(f"_{label}_")
                    st.image(
                        source_image,
                        width="stretch",
                        alt=f"DIP step {step_number} {label.lower()} view for {title}",
                    )

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
        st.image(
            image,
            caption="Uploaded X-ray",
            width="stretch",
            alt="Uploaded chest X-ray",
        )
    with chart_column:
        st.subheader("Predicted disease probabilities")
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.barh(class_names[::-1], scores[::-1], color="#3182bd")
        axis.set_xlim(0, 1)
        axis.set_xlabel("Probability")
        axis.set_ylabel("Disease class")
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        st.pyplot(
            figure,
            clear_figure=True,
            alt="Bar chart of model scores for 14 chest X-ray disease classes",
        )
        plt.close(figure)

    if st.toggle("Generate Grad-CAM explanation", value=False):
        selected_class = st.selectbox("Grad-CAM class", class_names, index=0)
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
            width="stretch",
            alt="Grad-CAM heatmap overlay on the uploaded chest X-ray",
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
