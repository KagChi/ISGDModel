import csv
import re
import unicodedata
import os
import glob

# Normalize text: remove accents, symbols, lowercase, etc.
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Load raw comments and save labeled data
def auto_label(input_csv, output_csv):
    all_labeled = []
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = normalize_text(row['text'])
            all_labeled.append({'text': text, 'label': row['label']})

    print(f"✅ Labeled comments from {input_csv}")
    return all_labeled

if __name__ == "__main__":
    # Use glob to find all CSV files in the "csv/comments" directory
    input_files = glob.glob("csv/dataset/*.csv")

    # Create the "csv/dataset" directory if it doesn't exist
    output_dir = "csv/normalized"
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        # Generate the output file name based on the input file name
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.csv")

        # Delete the output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # Auto label the data and save it to the output file
        labeled_data = auto_label(input_file, output_file)

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label'])
            writer.writeheader()
            writer.writerows(labeled_data)

        print(f"✅ Labeled {len(labeled_data)} comments → saved to: {output_file}")
