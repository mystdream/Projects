import streamlit as st
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import cv2
import numpy as np
import uuid
import sqlite3
import clip
import easyocr
import json
import pandas as pd
import matplotlib.pyplot as plt
import io

# Import necessary functions from your existing code
# (You'll need to make sure these functions are properly defined)
from your_module import (
    load_model, preprocess_image, segment_image, visualize_segmentation,
    extract_objects, save_objects, create_database, store_metadata,
    load_clip_model, identify_object, process_objects, extract_text,
    generate_summary, map_data, annotate_image, create_summary_table
)

def main():
    st.title("Image Analysis App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process button
        if st.button('Process Image'):
            # Step 1: Image Segmentation
            st.subheader("Step 1: Image Segmentation")
            model = load_model()
            image_tensor = preprocess_image(uploaded_file)
            prediction = segment_image(model, image_tensor)
            masks, scores = prediction['masks'], prediction['scores']

            # Visualize segmentation
            fig = visualize_segmentation(image_tensor, masks, scores)
            st.pyplot(fig)

            # Step 2: Object Extraction
            st.subheader("Step 2: Object Extraction")
            extracted_objects = extract_objects(uploaded_file, masks, scores)
            object_ids = save_objects(extracted_objects, "extracted_objects")

            # Create database and store metadata
            conn = create_database()
            master_id = str(uuid.uuid4())
            store_metadata(conn, object_ids, master_id)
            conn.close()

            st.write(f"Extracted {len(object_ids)} objects. Master ID: {master_id}")

            # Step 3: Object Identification
            st.subheader("Step 3: Object Identification")
            clip_model, preprocess, device = load_clip_model()
            object_descriptions = process_objects("extracted_objects", clip_model, preprocess, device)
            
            # Display object descriptions
            for obj_id, desc in object_descriptions.items():
                st.write(f"Object {obj_id}: {desc['description']}")

            # Step 4: Text Extraction
            st.subheader("Step 4: Text Extraction")
            reader = easyocr.Reader(['en'])
            object_text_data = {}
            for obj_id in object_ids:
                image_path = f"extracted_objects/{obj_id}.png"
                extracted_data = extract_text(reader, image_path)
                object_text_data[obj_id] = extracted_data
                
                # Display extracted text
                st.write(f"Object {obj_id} text: {' '.join([item['text'] for item in extracted_data])}")

            # Step 5: Object Summarization
            st.subheader("Step 5: Object Summarization")
            object_summaries = {}
            for obj_id in object_ids:
                summary = generate_summary(obj_id, object_descriptions[obj_id], object_text_data[obj_id])
                object_summaries[obj_id] = summary
                st.write(f"Object {obj_id} summary: {summary}")

            # Step 6: Data Mapping
            st.subheader("Step 6: Data Mapping")
            object_metadata = {obj_id: {"master_id": master_id} for obj_id in object_ids}
            mapped_data = map_data(object_metadata, object_descriptions, object_text_data, object_summaries)
            
            # Display mapped data
            st.json(mapped_data)

            # Step 7: Visualization and Reporting
            st.subheader("Step 7: Visualization and Reporting")
            annotated_image = annotate_image(np.array(image), mapped_data[master_id])
            st.image(annotated_image, caption='Annotated Image', use_column_width=True)

            summary_table = create_summary_table(mapped_data[master_id])
            st.dataframe(summary_table)

            # Option to download results
            csv = summary_table.to_csv(index=False)
            st.download_button(
                label="Download summary table as CSV",
                data=csv,
                file_name="summary_table.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()