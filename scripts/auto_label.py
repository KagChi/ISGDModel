import csv
import re
import unicodedata
import os
import glob

# Keywords commonly found in online gambling spam
GAMBLING_KEYWORDS = [
    "pulauwin", "dora77", "daftar", "dra77", "dora", "slot77", "pr0be855", "p r o b e t 8 5 5", "axl777", "hoki777",
    "maxwin", "deposit", "wd", "rtp", "jackpot", "jp", "alexis17", "weton88", "p l u t o 8 8", "d77", "p u l a u w i n", "luna p l a y 88", "maxwin",
    "a e r o 8 8", "aero 88", "ae r o 8 8", "ERO88", "cuan328", "g a c 000 r", "g4c0r", "alexa22", "weton88", "mona4d", "kusumat0t0", "squad777", "aero88",
    "probt855", "sgi88", "pstoto99", "777", "pulau777", "ula777", "jepey", "berkah99", "alexis 17", "manjurbet", "k o i s l o t", "m i y a 88", "layla 88",
    "dwadoa", "dwadora", "dewdr", "dwado", "ga ru da ho ki", "ero88", "thor311", "jepee", "doa77", "wedeey", "a e r o 88", "A E R O DELAPAN DELAPAN"
]

# Normalize text: remove accents, symbols, lowercase, etc.
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Check if any gambling keyword appears in the comment
def is_gambling_comment(text):
    for keyword in GAMBLING_KEYWORDS:
        if keyword in text:
            return 1
    return 0

# Load raw comments and save labeled data
def auto_label(input_csv, output_csv):
    all_labeled = []
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            label = is_gambling_comment(normalize_text(text))

            # all_labeled.append({'text': text, 'label': label})

            if label == 1:
                all_labeled.append({'text': text, 'label': label})

    print(f"✅ Labeled comments from {input_csv}")
    return all_labeled

if __name__ == "__main__":
    # Use glob to find all CSV files in the "csv/comments" directory
    input_files = glob.glob("csv/comments/*.csv")

    # Create the "csv/dataset" directory if it doesn't exist
    output_dir = "csv/flagged"
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
