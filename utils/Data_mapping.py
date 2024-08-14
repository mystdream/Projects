import json
import sqlite3
import os

def load_data_from_previous_steps():
    # Load object descriptions (Step 3)
    with open('object_descriptions.json', 'r') as f:
        object_descriptions = json.load(f)

    # Load text data (Step 4)
    with open('object_text_data.json', 'r') as f:
        object_text_data = json.load(f)

    # Load object summaries (Step 5)
    with open('object_summaries.json', 'r') as f:
        object_summaries = json.load(f)

    return object_descriptions, object_text_data, object_summaries

def get_object_metadata_from_db():
    conn = sqlite3.connect('objects_database.db')
    c = conn.cursor()
    c.execute("SELECT id, master_id FROM objects")
    object_metadata = {row[0]: {"master_id": row[1]} for row in c.fetchall()}
    conn.close()
    return object_metadata

def map_data(object_metadata, object_descriptions, object_text_data, object_summaries):
    mapped_data = {}

    for object_id, metadata in object_metadata.items():
        master_id = metadata['master_id']

        if master_id not in mapped_data:
            mapped_data[master_id] = {
                "objects": {}
            }

        mapped_data[master_id]["objects"][object_id] = {
            "identification": object_descriptions.get(object_id, {}),
            "extracted_text": object_text_data.get(object_id, []),
            "summary": object_summaries.get(object_id, "")
        }

    return mapped_data

def save_mapped_data(mapped_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(mapped_data, f, indent=2)

def main():
    # Load data from previous steps
    object_descriptions, object_text_data, object_summaries = load_data_from_previous_steps()

    # Get object metadata from the database
    object_metadata = get_object_metadata_from_db()

    # Map all data
    mapped_data = map_data(object_metadata, object_descriptions, object_text_data, object_summaries)

    # Save the mapped data
    output_file = "mapped_data.json"
    save_mapped_data(mapped_data, output_file)

    print(f"Data mapping complete. Mapped data saved to {output_file}")

    return mapped_data

if __name__ == "__main__":
    main()