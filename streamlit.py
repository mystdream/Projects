import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gc

# Force CPU usage
device = torch.device('cpu')

# Load the model
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image, max_size=800):
    w, h = image.size
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h))
    image_tensor = F.to_tensor(image).to(device)
    return image_tensor.unsqueeze(0)

# Segment the image
def segment_image(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    return prediction

# Visualize segmentation
def visualize_segmentation(image, boxes, scores, threshold=0.5):
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, f'{score:.2f}', bbox=dict(facecolor='white', alpha=0.8))
    ax.axis('off')
    plt.tight_layout()
    return fig

# Streamlit app
def main():
    st.title("Object Detection and Contextual Analysis")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Detecting objects...")
        image_tensor = preprocess_image(image)
        prediction = segment_image(model, image_tensor)
        
        boxes = prediction['boxes']
        scores = prediction['scores']
        
        st.write("Detected objects with confidence scores:")
        for i, score in enumerate(scores):
            if score > 0.5:
                st.write(f"Object {i+1}: Confidence {score:.2f}")
        
        fig = visualize_segmentation(image_tensor, boxes, scores)
        st.pyplot(fig)
        
        # Clear memory
        del image_tensor, prediction, boxes, scores
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
