import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_previous_data(identification_file, text_data_file):
    with open(identification_file, 'r') as f:
        identification_data = json.load(f)

    with open(text_data_file, 'r') as f:
        text_data = json.load(f)

    return identification_data, text_data

def extract_key_terms(text, n=5):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in word_tokens if w.isalnum() and w not in stop_words]

    # Count and return top N most common terms
    return [word for word, _ in Counter(filtered_tokens).most_common(n)]

def generate_summary(object_id, identification, text_data):
    top_category = identification['top_categories'][0]['category']
    confidence = identification['top_categories'][0]['confidence']

    summary = f"Object {object_id} is identified as a {top_category} with {confidence:.2f} confidence. "

    if len(identification['top_categories']) > 1:
        second_category = identification['top_categories'][1]['category']
        summary += f"It might also be a {second_category}. "

    if text_data:
        extracted_text = ' '.join([item['text'] for item in text_data])
        key_terms = extract_key_terms(extracted_text)
        if key_terms:
            summary += f"Key terms associated with this object are: {', '.join(key_terms)}. "
    else:
        summary += "No text was extracted from this object. "

    return summary.strip()

def process_objects(identification_data, text_data):
    object_summaries = {}

    for object_id in identification_data.keys():
        identification = identification_data[object_id]
        object_text_data = text_data.get(object_id, [])

        summary = generate_summary(object_id, identification, object_text_data)
        object_summaries[object_id] = summary

    return object_summaries

def save_summaries(object_summaries, output_file):
    with open(output_file, 'w') as f:
        json.dump(object_summaries, f, indent=2)

def main(identification_file, text_data_file):
    # Load data from previous steps
    identification_data, text_data = load_previous_data(identification_file, text_data_file)

    # Process all objects
    object_summaries = process_objects(identification_data, text_data)

    # Save the summaries
    output_file = "object_summaries.json"
    save_summaries(object_summaries, output_file)

    print(f"Object summarization complete. Summaries saved to {output_file}")

    return object_summaries

if __name__ == "__main__":
    identification_file = "object_descriptions.json"  # Output from Step 3
    text_data_file = "object_text_data.json"  # Output from Step 4
    main(identification_file, text_data_file)

