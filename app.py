import streamlit as st
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2
import os
import uuid
import json
from collections import Counter
import easyocr
import clip
import nltk
import sqlite3

# Load necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the models
@st.cache_resource
def load_models():
    segment_model = maskrcnn_resnet50_fpn(pretrained=True)
    segment_model.eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    ocr_reader = easyocr.Reader(['en'])
    return segment_model, clip_model, clip_preprocess, ocr_reader

# Define functions for image processing, segmentation, and analysis (same as your previous functions)
def preprocess_image(image):
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor

def segment_image(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    return prediction

def extract_objects(image, masks, scores, threshold=0.5):
    image_np = np.array(image)
    extracted_objects = []
    for mask, score in zip(masks, scores):
        if score > threshold:
            binary_mask = (mask.squeeze().numpy() > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            object_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            cv2.drawContours(object_mask, contours, -1, (255), thickness=cv2.FILLED)
            extracted_object = cv2.bitwise_and(image_np, image_np, mask=object_mask)
            x, y, w, h = cv2.boundingRect(object_mask)
            cropped_object = extracted_object[y:y+h, x:x+w]
            extracted_objects.append(cropped_object)
    return extracted_objects

def identify_object(clip_model, clip_preprocess, image, device):
    image = clip_preprocess(image).unsqueeze(0).to(device)
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                  "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    text = clip.tokenize(categories).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
    results = [{"category": categories[idx], "confidence": value.item()} for value, idx in zip(values, indices)]
    description = f"This image appears to contain a {results[0]['category']}. "
    if len(results) > 1:
        description += f"It might also be a {results[1]['category']} or a {results[2]['category']}."
    return results, description

# Streamlit application code
def main():
    st.title("AI Image Segmentation and Object Analysis")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        segment_model, clip_model, clip_preprocess, ocr_reader = load_models()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        st.write("Segmenting the image...")
        image_tensor = preprocess_image(image)
        prediction = segment_image(segment_model, image_tensor)
        masks = prediction['masks']
        scores = prediction['scores']

        extracted_objects = extract_objects(image, masks, scores)
        
        st.write("Analyzing extracted objects...")
        object_descriptions = {}
        for idx, obj in enumerate(extracted_objects):
            obj_id = str(uuid.uuid4())
            obj_pil = Image.fromarray(obj)
            results, description = identify_object(clip_model, clip_preprocess, obj_pil, device)
            object_descriptions[obj_id] = {"top_categories": results, "description": description}
            st.image(obj_pil, caption=f'Extracted Object {idx+1}', use_column_width=True)
            st.write(description)

        # Saving descriptions (optional, for further steps)
        with open('object_descriptions.json', 'w') as f:
            json.dump(object_descriptions, f, indent=2)

        st.write("Analysis complete.")

if __name__ == "__main__":
    main()
