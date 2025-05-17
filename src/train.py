import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
import pickle

def load_csvs(files):
    all_data = []
    for file_path in files:
        data = pd.read_csv(file_path, header=None, names=['text', 'label'])
        data = data[data['label'].astype(str).str.strip().isin(['0','1'])]
        data['label'] = data['label'].astype(int)
        data = data.dropna(subset=['text'])
        for _, row in data.iterrows():
            all_data.append((row['text'], row['label']))
    return all_data

# Load dataset
paths = ['csv/dataset/*.csv', 'csv/normalized/*.csv']
filecsv = []
for path in paths:
    filecsv.extend(glob.glob(path))

data_list = []
for file in filecsv:
    data = pd.read_csv(file, header=None, names=['text', 'label'])
    data = data[data['label'].astype(str).str.strip().isin(['0','1'])]
    data['label'] = data['label'].astype(int)
    data = data.dropna(subset=['text'])
    data_list.append(data)

data = pd.concat(data_list, ignore_index=True)

# Split data train-test
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer and IndoBERT-base-p1 model
model_name = "indobenchmark/indobert-base-p1"
model_path = "./tmp/models/v2"  # Path to the local model

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Loaded tokenizer and model from local path:", model_path)
except Exception as e:
    print(f"Failed to load local model from {model_path}. Loading from Hugging Face Model Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print("Loaded tokenizer and model from Hugging Face Model Hub:", model_name)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    return acc

# Training arguments
training_args = TrainingArguments(
    output_dir="tmp/models/v2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save to hungingface format
model.save_pretrained("tmp/models/v2")
tokenizer.save_pretrained("tmp/models/v2")

# Save model to .pkl format
with open("models/v2/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/v2/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
