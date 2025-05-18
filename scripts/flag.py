import csv
import os
import glob

def load_predict_texts(predict_folder):
    predict_texts = set()
    predict_files = glob.glob(os.path.join(predict_folder, '*.csv'))
    for pf in predict_files:
        with open(pf, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predict_texts.add(row['text'])
    print(f"Loaded {len(predict_texts)} unique texts from predict folder")
    return predict_texts

def flag_dataset_inplace(dataset_folder, predict_texts):
    dataset_files = glob.glob(os.path.join(dataset_folder, '*.csv'))

    for file in dataset_files:
        labeled_data = []
        with open(file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row['text']
                label = '1' if text in predict_texts else row['label']
                labeled_data.append({'text': text, 'label': label})

        with open(file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=['text', 'label'])
            writer.writeheader()
            writer.writerows(labeled_data)

        print(f"✅ Updated labels in {file}")

if __name__ == "__main__":
    predict_folder = "csv/flag"
    dataset_folder = "csv/dataset"

    predict_texts = load_predict_texts(predict_folder)
    flag_dataset_inplace(dataset_folder, predict_texts)
