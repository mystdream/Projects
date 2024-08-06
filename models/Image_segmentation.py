import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0)

def segment_image(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    return prediction

def visualize_segmentation(image, masks, scores, threshold=0.5):
    image = image.squeeze().permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    for mask, score in zip(masks, scores):
        if score > threshold:
            masked = np.where(mask.squeeze().numpy() > 0.5, 1, 0)
            plt.contour(masked, colors=['red'], alpha=0.5, linewidths=2)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('segmented_image.png')
    plt.close()

def main(image_path):
    model = load_model()
    image_tensor = preprocess_image(image_path)
    prediction = segment_image(model, image_tensor)

    masks = prediction['masks']
    scores = prediction['scores']

    visualize_segmentation(image_tensor, masks, scores)
    print(f"Segmentation complete. Output saved as 'segmented_image.png'")
    return masks, scores

if __name__ == "__main__":
    image_path = "/content/WhatsApp Image 2024-02-27 at 3.07.59 PM (2).jpeg"
    main(image_path)