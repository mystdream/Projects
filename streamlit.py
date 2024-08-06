import streamlit as st
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0)

# Segment the image
def segment_image(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    return prediction

# Visualize segmentation
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
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Object Detection and Contextual Analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Detecting objects...")

        model = load_model()
        image_tensor = preprocess_image(image)
        prediction = segment_image(model, image_tensor)

        masks = prediction['masks']
        scores = prediction['scores']

        st.write("Detected objects with confidence scores:")
        for i, score in enumerate(scores):
            if score > 0.5:
                st.write(f"Object {i+1}: Confidence {score:.2f}")

        visualize_segmentation(image_tensor, masks, scores)

if __name__ == "__main__":
    main()
