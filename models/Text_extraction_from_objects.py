import easyocr
import os
import json
from PIL import Image
import numpy as np

def load_ocr_reader(languages=['en']):
    return easyocr.Reader(languages)

def extract_text(reader, image_path):
    # Read the image
    image = Image.open(image_path)
    image_np = np.array(image)

    # Perform OCR
    results = reader.readtext(image_np)

    # Extract text and confidence
    extracted_data = [
        {
            "text": result[1],
            "confidence": result[2],
            "bounding_box": result[0]
        } for result in results
    ]

    return extracted_data

def process_objects(input_folder, reader):
    object_text_data = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            object_id = os.path.splitext(filename)[0]
            image_path = os.path.join(input_folder, filename)

            extracted_data = extract_text(reader, image_path)

            object_text_data[object_id] = extracted_data

    return object_text_data

def save_text_data(object_text_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(object_text_data, f, indent=2)

def main(input_folder):
    # Load the OCR reader
    reader = load_ocr_reader()

    # Process all objects
    object_text_data = process_objects(input_folder, reader)

    # Save the extracted text data
    output_file = "object_text_data.json"
    save_text_data(object_text_data, output_file)

    print(f"Text extraction complete. Data saved to {output_file}")

    return object_text_data

if __name__ == "__main__":
    input_folder = "extracted_objects"  # Folder containing extracted object images from Step 2
    main(input_folder)

