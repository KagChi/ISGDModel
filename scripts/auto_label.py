import csv
import re
import unicodedata
import os
import glob
import emoji

# Keywords commonly found in online gambling spam
GAMBLING_KEYWORDS = [
    "pulauwin", "dora77", "daftar", "dra77", "dora", "slot77", "pr0be855", "p r o b e t 8 5 5", "axl777", "hoki777",
    "maxwin", "deposit", "wd", "rtp", "alexis17", "weton88", "p l u t o 8 8", "d77", "p u l a u w i n", "luna p l a y 88", "maxwin",
    "a e r o 8 8", "aero 88", "ae r o 8 8", "ERO88", "cuan328", "g a c 000 r", "g4c0r", "alexa22", "weton88", "mona4d", "kusumat0t0", "squad777", "aero88",
    "probt855", "sgi88", "pstoto99", "pulau777", "ula777", "jepey", "berkah99", "alexis 17", "manjurbet", "k o i s l o t", "k oislot", "m i y a 88", "layla 88",
    "dwadoa", "dwadora", "dewdr", "dwado", "ga ru da ho ki", "ero88", "thor311", "jepee", "doa77", "wedeey", "a e r o 88", "A E R O DELAPAN DELAPAN",
    "lesti77", "jet88bet", "ayamwin", "zoom555", "ringbet88", "momo99 99", "lohanslt", "neng4d", "n e n g 4 d", "r a d a r 138", "poa88", "poa 88",
    "a l e x i s 1 7", "y u k     6     9", "mgs88", "dewador", "dewadora", "asiagenting", "sgi88", "sijago 88", "pr0be855"
]

def contains_emoji(text):
    return bool(emoji.replace_emoji(text, replace='').strip() != text.strip())

def contains_gambling_keyword(text):
    return contains_emoji(text) and "17" in text;

# Normalize text: remove accents, symbols, lowercase, etc.
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def clean_text(text, remove_emoji=True):
    text = re.sub(r'<a\s+href="[^"]*">.*?</a>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@{1,2}[^\s]+', '', text)

    if remove_emoji and not contains_gambling_keyword(text):
        text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('"', '')

    return text


# Check if any gambling keyword appears in the comment
def is_gambling_comment(text):
    if contains_gambling_keyword(text):
        return 1
    
    for keyword in GAMBLING_KEYWORDS:
        if keyword in normalize_text(text):
            return 1

    return 0

# Load raw comments and save labeled data
def auto_label(input_csv, output_csv):
    all_labeled = []
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            cleaned_text = clean_text(text)
            label = is_gambling_comment(clean_text(text, False))

            all_labeled.append({'text': cleaned_text, 'label': label})

            # if label == 1:
            #     all_labeled.append({'text': text, 'label': label})

    print(f"✅ Labeled comments from {input_csv}")
    return all_labeled

if __name__ == "__main__":
    # Use glob to find all CSV files in the "csv/comments" directory
    input_files = glob.glob("csv/comments/*.csv")

    # Create the "csv/dataset" directory if it doesn't exist
    output_dir = "csv/dataset"
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
