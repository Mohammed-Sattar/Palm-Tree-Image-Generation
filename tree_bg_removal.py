"""
Removing the background from tree images using a segmentation model.
Key Components:
    Model Initialization:
        Loads the DeepLabV3 segmentation model pre-trained on a dataset.
    Preprocessing Function:
        Prepares images for segmentation by resizing, normalizing, and converting them to tensors.
    Post-processing Function:
        Processes the output from the segmentation model to create a mask and visualize the segmentation results.
"""

import torch
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.utils import draw_segmentation_masks
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the DeepLabv3 model with updated weights parameter
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet101(weights=weights)
model.eval()

# Preprocessing function
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

# Post-processing function
def postprocess(output, original_image):
    mask = output['out'][0].argmax(0).byte().cpu().numpy()

    # Debugging: Visualize the raw segmentation mask
    plt.imshow(mask, cmap="nipy_spectral")
    plt.title("Raw Segmentation Mask")
    plt.colorbar()
    plt.show()

    # Ensure `original_image` is a PIL Image
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)

    # Resize the mask to match the original image size
    mask_resized = Image.fromarray(mask).resize(original_image.size, resample=Image.NEAREST)
    mask_resized = np.array(mask_resized)

    # Debugging: Print unique class IDs in the mask
    print("Unique classes in the mask:", np.unique(mask_resized))

    # Create a blank white image
    white_background = np.ones_like(np.array(original_image)) * 255

    # Adjust class ID for trees (if needed, based on debugging)
    target_class_id = 15  # Update this if the debugging indicates a different class ID

    # Apply the mask: keep target regions and make the rest white
    segmented = np.where(mask_resized[..., None] == target_class_id, np.array(original_image), white_background)
    return segmented



# Main function
def segment_tree(image_path):
    input_image, original_image = preprocess(image_path)
    with torch.no_grad():
        output = model(input_image)
    segmented_image = postprocess(output, np.array(original_image))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()
    return segmented_image

# Segment the uploaded image
segment_tree(r"Code\Tree Age Predictor & Generator\tree_images\17_years\JT2001_1.jpg")
