import torch
from PIL import Image
import clip
import os
import json

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def identify_object(model, preprocess, image_path, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # List of potential object categories
    categories = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    text = clip.tokenize(categories).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

    results = [
        {"category": categories[idx], "confidence": value.item()}
        for value, idx in zip(values, indices)
    ]

    # Generate a description
    top_category = results[0]["category"]
    description = f"This image appears to contain a {top_category}. "
    if len(results) > 1:
        description += f"It might also be a {results[1]['category']} or a {results[2]['category']}."

    return results, description

def process_objects(input_folder, model, preprocess, device):
    object_descriptions = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            object_id = filename[:-4]  # Remove .png extension
            image_path = os.path.join(input_folder, filename)

            results, description = identify_object(model, preprocess, image_path, device)

            object_descriptions[object_id] = {
                "top_categories": results,
                "description": description
            }

    return object_descriptions

def save_descriptions(object_descriptions, output_file):
    with open(output_file, 'w') as f:
        json.dump(object_descriptions, f, indent=2)

def main(input_folder):
    model, preprocess, device = load_clip_model()
    object_descriptions = process_objects(input_folder, model, preprocess, device)

    output_file = "object_descriptions.json"
    save_descriptions(object_descriptions, output_file)

    print(f"Object identification complete. Descriptions saved to {output_file}")

if __name__ == "__main__":
    input_folder = "extracted_objects"  # Folder containing extracted object images from Step 2
    main(input_folder)

