import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def load_mapped_data(mapped_data_file):
    with open(mapped_data_file, 'r') as f:
        return json.load(f)

def load_image(image_path):
    return cv2.imread(image_path)

def annotate_image(image, objects_data):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Use a default font
    font = ImageFont.load_default()

    for obj_id, obj_data in objects_data['objects'].items():
        # Assuming bounding box information is available in obj_data
        # If not, you might need to adjust this part
        bbox = obj_data.get('bounding_box', [0, 0, 100, 100])  # default values if not available
        x, y, w, h = bbox

        # Draw bounding box
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)

        # Draw label
        label = obj_data['identification']['top_categories'][0]['category']
        draw.text((x, y-20), f"{label} ({obj_id})", font=font, fill="red")

    return np.array(pil_image)

def create_summary_table(objects_data):
    rows = []
    for obj_id, obj_data in objects_data['objects'].items():
        row = {
            'Object ID': obj_id,
            'Category': obj_data['identification']['top_categories'][0]['category'],
            'Confidence': obj_data['identification']['top_categories'][0]['confidence'],
            'Extracted Text': '; '.join([text['text'] for text in obj_data['extracted_text']]),
            'Summary': obj_data['summary']
        }
        rows.append(row)

    return pd.DataFrame(rows)

def save_output(annotated_image, summary_table, output_image_path, output_table_path):
    # Save annotated image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

    # Save summary table
    summary_table.to_csv(output_table_path, index=False)

def main(mapped_data_file, original_image_path):
    # Load mapped data
    mapped_data = load_mapped_data(mapped_data_file)

    # Process each master image
    for master_id, master_data in mapped_data.items():
        # Load original image
        original_image = load_image(original_image_path)

        # Annotate image
        annotated_image = annotate_image(original_image, master_data)

        # Create summary table
        summary_table = create_summary_table(master_data)

        # Save outputs
        output_image_path = f"annotated_image_{master_id}.png"
        output_table_path = f"summary_table_{master_id}.csv"
        save_output(annotated_image, summary_table, output_image_path, output_table_path)

        print(f"Output generated for master image {master_id}:")
        print(f"- Annotated image saved as {output_image_path}")
        print(f"- Summary table saved as {output_table_path}")

if __name__ == "__main__":
    mapped_data_file = "mapped_data.json"
    original_image_path = "/content/WhatsApp Image 2024-02-27 at 3.07.59 PM (2).jpeg"  # Replace with actual path
    main(mapped_data_file, original_image_path)

