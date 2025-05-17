import csv
import re
import unicodedata
import os
import glob
import emoji

# Keywords commonly found in online gambling spam
GAMBLING_PATTERNS = [
    r"p\s*u\s*l\s*a\s*u\s*w\s*i\s*n", 
    r"d\s*o\s*r\s*a\s*7\s*7", 
    r"alexis\s*1\s*7", 
    r"a\s*l\s*e\s*x\s*i\s*s\s*1\s*7", 
    r"weton\s*88", 
    r"p\s*l\s*u\s*t\s*o\s*8\s*8", 
    r"slot\s*77", 
    r"probe\s*855", 
    r"p\s*r\s*o\s*b\s*e\s*t\s*8\s*5\s*5", 
    r"a\s*e\s*r\s*o\s*8\s*8", 
    r"aero\s*88", 
    r"ae\s*r\s*o\s*8\s*8", 
    r"ero\s*88", 
    r"g\s*a\s*c\s*0\s*0\s*0\s*r", 
    r"g\s*4\s*c\s*0\s*r", 
    r"koislot", 
    r"m\s*i\s*y\s*a\s*8\s*8", 
    r"layla\s*88", 
    r"dwadoa", r"dwadora", r"dewadora", r"dewador", 
    r"y\s*u\s*k\s*6\s*9",
    r"maxwin", r"deposit", r"wd", r"rtp", r"jackpot", r"\bjp\b",
    r"777", r"jet88", r"zoom555", r"ringbet88", r"thor311", r"cuan328", r"mona4d", r"ayamwin",
    r"poker", r"slot", r"casino", r"judi", r"judi online", r"judi slot",
    r"poa\s*88", r"pr0be\s*8\s*5\s*5", r"sijago\s*88", r"k\s*o\s*i\s*s\s*l\s*o\s*t",
    r"asiagenting", r"sgi88"
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

GAMBLING_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in GAMBLING_PATTERNS]

# Check if any gambling keyword appears in the comment
def is_gambling_comment(text):
    if contains_gambling_keyword(text):
        return 1
    
    text_norm = normalize_text(text)
    for regex in GAMBLING_REGEXES:
        if regex.search(text_norm):
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
