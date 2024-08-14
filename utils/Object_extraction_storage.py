import cv2
import numpy as np
from PIL import Image
import sqlite3
import os
import uuid
import torch

def extract_objects(image_path, masks, scores, threshold=0.5):
    # Load the original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    extracted_objects = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score > threshold:
            # Convert mask to binary
            binary_mask = (mask.squeeze().numpy() > 0.5).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask for the largest contour
            object_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            cv2.drawContours(object_mask, contours, -1, (255), thickness=cv2.FILLED)

            # Extract the object
            extracted_object = cv2.bitwise_and(image_rgb, image_rgb, mask=object_mask)

            # Crop the object to its bounding box
            x, y, w, h = cv2.boundingRect(object_mask)
            cropped_object = extracted_object[y:y+h, x:x+w]

            extracted_objects.append(cropped_object)

    return extracted_objects

def save_objects(extracted_objects, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    object_ids = []
    for i, obj in enumerate(extracted_objects):
        obj_id = str(uuid.uuid4())  # Generate a unique ID
        object_ids.append(obj_id)

        # Convert from BGR to RGB
        obj_rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)

        # Save the object as an image
        img = Image.fromarray(obj_rgb)
        img.save(os.path.join(output_folder, f"{obj_id}.png"))

    return object_ids

def create_database():
    conn = sqlite3.connect('objects_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS objects
                 (id TEXT PRIMARY KEY, master_id TEXT)''')
    conn.commit()
    return conn

def store_metadata(conn, object_ids, master_id):
    c = conn.cursor()
    for obj_id in object_ids:
        c.execute("INSERT INTO objects (id, master_id) VALUES (?, ?)", (obj_id, master_id))
    conn.commit()

def main(image_path, model):
    # Use the model to get masks and scores
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    masks = prediction['masks']
    scores = prediction['scores']

    # Extract objects
    extracted_objects = extract_objects(image_path, masks, scores)

    # Save objects and get their IDs
    output_folder = "extracted_objects"
    object_ids = save_objects(extracted_objects, output_folder)

    # Generate a master ID for the original image
    master_id = str(uuid.uuid4())

    # Store metadata in the database
    conn = create_database()
    store_metadata(conn, object_ids, master_id)
    conn.close()

    print(f"Extracted {len(object_ids)} objects. Master ID: {master_id}")
    return object_ids, master_id

if __name__ == "__main__":
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    image_path = "/content/WhatsApp Image 2024-02-27 at 3.07.59 PM (2).jpeg"
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    main(image_path, model)
