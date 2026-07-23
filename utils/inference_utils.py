"""
inference_utils.py

Shared utilities for loading a trained classifier (from B3a) and running
inference with it -- either against sampled dataset images, or live against
a camera feed.
"""

import os
import json
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from train_utils import build_model, list_class_names, list_image_files, DEFAULT_MODELS_DIR
from jupyter_utils import register_observer, unregister_observer

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def _to_pil(image):
    """Converts either a PIL Image or a BGR numpy array (as produced by
    the camera) into a PIL Image, so predict_image() can accept either.

    Args:
        image (PIL.Image.Image or numpy.ndarray): the image to convert

    Returns:
        PIL.Image.Image
    """
    if isinstance(image, Image.Image):
        return image

    # Assume a BGR numpy array (OpenCV/camera convention) -- flip to RGB
    return Image.fromarray(image[:, :, ::-1])


def load_model_and_metadata(model_path, models_dir=DEFAULT_MODELS_DIR):
    """Loads a trained model checkpoint along with the training record that
    describes how it was trained, by matching the checkpoint's model_name
    against entries in training_log.txt.

    Args:
        model_path (str): path to a .pth checkpoint (e.g. from
            ipyfilechooser), expected to be named "best_model_<name>.pth"
        models_dir (str): directory containing training_log.txt

    Returns:
        (torch.nn.Module, torch.device, list[str], dict): the loaded model
            (in eval mode), its device, its class_names, and the full
            training record logged for it in B3a

    Raises:
        FileNotFoundError: if no matching entry is found in training_log.txt
    """
    filename = os.path.basename(model_path)
    model_name = filename
    if model_name.startswith('best_model_'):
        model_name = model_name[len('best_model_'):]
    if model_name.endswith('.pth'):
        model_name = model_name[:-len('.pth')]

    log_path = os.path.join(models_dir, "training_log.txt")
    training_record = None

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record.get('model_name') == model_name:
                    training_record = record  # keep the most recent matching entry

    if training_record is None:
        raise FileNotFoundError(
            f"No training record found for model_name '{model_name}' in {log_path}. "
            "Make sure you selected a model that was trained in B3a."
        )

    class_names = training_record['class_names']

    model, device = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loaded '{model_name}' -- classes: {class_names}")
    print(
        f"Trained on {training_record['num_train_images']} images, "
        f"best test accuracy: {training_record['best_test_accuracy']:.1%}"
    )

    return model, device, class_names, training_record


def predict_image(model, image, class_names, device):
    """Runs a single image through `model` and returns its predicted class
    and confidence.

    Args:
        model (torch.nn.Module): a loaded, eval-mode model
        image (PIL.Image.Image or numpy.ndarray): the image to classify
        class_names (list[str]): class names in the model's output order
        device (torch.device): device the model lives on

    Returns:
        (str, float): predicted class name, and the model's confidence
            in that prediction (0.0-1.0)
    """
    pil_image = _to_pil(image)
    tensor = INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities))
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_label, confidence


def show_inference_grid(model, class_names, device, data_dir, num_images=8, grid_shape=(2, 4)):
    """Randomly samples images from the dataset and displays them in a
    grid with their true label and the model's prediction underneath --
    green if correct, red if incorrect. Re-run this cell for a new
    random sample.

    Note: this workshop doesn't set aside a held-out test set for this
    step, so some (or all) of these images may have been part of what the
    model trained on. Strong performance here isn't the same guarantee of
    generalization as the test accuracy reported during training in B3a.

    Args:
        model (torch.nn.Module): a loaded, eval-mode model
        class_names (list[str]): class names in the model's output order
        device (torch.device): device the model lives on
        data_dir (str): dataset root to sample images from
        num_images (int): how many images to sample and classify
        grid_shape (tuple[int, int]): (rows, cols) for the display grid;
            rows * cols must equal num_images

    Returns:
        None
    """
    rows, cols = grid_shape
    if rows * cols != num_images:
        raise ValueError(f"grid_shape {grid_shape} doesn't hold num_images={num_images} images.")

    all_images = []
    for class_name in list_class_names(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        for filename in list_image_files(class_dir):
            all_images.append((os.path.join(class_dir, filename), class_name))

    if len(all_images) < num_images:
        raise ValueError(
            f"Only found {len(all_images)} images in {data_dir}, need at least {num_images}."
        )

    sample = random.sample(all_images, num_images)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, (image_path, true_label) in zip(axes, sample):
        image = Image.open(image_path)
        predicted_label, confidence = predict_image(model, image, class_names, device)

        is_correct = predicted_label == true_label
        color = 'green' if is_correct else 'red'

        ax.imshow(image)
        ax.axis('off')
        ax.set_title(
            f"true: {true_label}\npredicted: {predicted_label} ({confidence:.0%})",
            color=color, fontsize=11
        )

    plt.tight_layout()
    plt.show()


def start_live_classification(camera, model, class_names, device, on_prediction):
    """Starts continuously classifying frames from `camera` as they
    arrive, calling `on_prediction(label, confidence)` on every new frame.

    Uses register_observer() (idempotent) so re-running this cell doesn't
    stack a second classification loop on top of the first.

    Args:
        camera (TraitletCamera): the live camera feed
        model (torch.nn.Module): a loaded, eval-mode model
        class_names (list[str]): class names in the model's output order
        device (torch.device): device the model lives on
        on_prediction (callable): called on every new frame as
            on_prediction(label, confidence) -- e.g. to update a Label
            widget's .value, recolor a status indicator, or anything else
            a particular notebook needs done with each new prediction

    Returns:
        None
    """
    def classify_frame(change):
        predicted_label, confidence = predict_image(model, change['new'], class_names, device)
        on_prediction(predicted_label, confidence)

    register_observer(camera, classify_frame, names='value')


def stop_live_classification(camera):
    """Stops continuous classification started by start_live_classification,
    if any is currently running.

    Args:
        camera (TraitletCamera): the live camera feed

    Returns:
        None
    """
    if unregister_observer(camera, names='value'):
        print("Live classification stopped.")
    else:
        print("Live classification wasn't running.")
