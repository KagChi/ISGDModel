import torch
import pickle
from transformers import AutoTokenizer
import pandas as pd
import logging
import glob
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Set True kalau mau lihat log tiap baris
log_each_row = False

try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    logger.info("Tokenizer loaded successfully.")

    logger.info("Loading model from pickle file...")

    original_torch_load = torch.load
    def torch_load_cpu(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = torch.device('cpu')
        return original_torch_load(*args, **kwargs)
    torch.load = torch_load_cpu

    with open("models/v2/model.pkl", "rb") as f:
        model = pickle.load(f)

    torch.load = original_torch_load  # restore original

    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
    else:
        logger.info("CUDA is not available. Using CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model moved to device: {device}")

    logger.info("Loading data from CSV...")
    csv_files = glob.glob("csv/flagged/*.csv")

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the csv/flagged/ directory")

    data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    logger.info(f"Data loaded successfully with {len(data)} rows from {len(csv_files)} files.")

    all_predictions = []
    all_labels = data['label'].tolist()
    prediction_labels = []
    prediction_texts = []
    correctness_flags = []

    max_text_len = 80

    for index, row in data.iterrows():
        text = row['text']
        label = row['label']

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        all_predictions.append(prediction)

        pred_text = 'judi' if prediction == 1 else 'bukan judi'
        label_text = 'judi' if label == 1 else 'bukan judi'
        correctness = 'Betul' if prediction == label else 'Salah prediksi'

        prediction_labels.append(prediction)
        prediction_texts.append(pred_text)
        correctness_flags.append(correctness)

        display_text = (text[:max_text_len] + '...') if len(text) > max_text_len else text

        if log_each_row:
            logger.info(f"[Row {index}] Text: \"{display_text}\" | Prediksi: {prediction} ({pred_text}), Label: {label} ({label_text}), Hasil: {correctness}")
            logger.info(f"+========================+")

    data['prediction'] = prediction_labels
    data['prediction_text'] = prediction_texts
    data['correctness'] = correctness_flags

    os.makedirs("csv/output", exist_ok=True)
    output_path = "csv/output/prediction_results.csv"
    data.to_csv(output_path, index=False)
    logger.info(f"Prediction results saved to {output_path}")

    # Hitung akurasi
    correct = sum([1 for p, l in zip(all_predictions, all_labels) if p == l])
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total} correct predictions)")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
