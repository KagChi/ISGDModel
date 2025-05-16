import csv
import re
import unicodedata
import os

# Keywords commonly found in online gambling spam
GAMBLING_KEYWORDS = [
    "pulauwin", "dora77", "daftar", "slot", "gacor", "dra77", "dora", "slot77", "pr0be855", "p r o b e t 8 5 5", "axl777", "hoki777",
    "maxwin", "deposit", "wd", "rtp", "jackpot", "spin", "jp", "alexis17", "weton88", "p l u t o 8 8", "d77", "p u l a u w i n", "luna p l a y 88"
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
    labeled = []
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            label = is_gambling_comment(normalize_text(text))
            normalized_text = normalize_text(text)
            is_gambling = is_gambling_comment(normalized_text)
            print(f"Text: {text}, Normalized: {normalized_text}, Is Gambling: {is_gambling}")
            labeled.append({'text': text, 'label': label})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(labeled)

    print(f"✅ Labeled {len(labeled)} comments → saved to: {output_csv}")

if __name__ == "__main__":
    auto_label("data/comments.csv", "data/new_labeled_comments.csv")
