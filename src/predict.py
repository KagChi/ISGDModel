import torch
import pickle
from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

with open("models/v2/model.pkl", "rb") as f:
    model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

data = pd.read_csv("csv/predicts.csv")

for index, row in data.iterrows():
    text = row['text']
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()

    print(f"Teks: {text}")
    print(f"Prediksi: {prediction} (0: Bukan Judi, 1: Judi)")
    print(f"Komentar ini terindikasi { 'judi' if prediction == 1 else 'bukan judi' }")
