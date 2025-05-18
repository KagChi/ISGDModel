import torch
import torch.nn.functional as F
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
    csv_files = glob.glob("csv/predict/*.csv")

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the csv/predict/ directory")

    data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    logger.info(f"Data loaded successfully with {len(data)} rows from {len(csv_files)} files.")

    label_counts = data['label'].value_counts().to_dict()
    judi_count = label_counts.get(1, 0)
    bukan_judi_count = label_counts.get(0, 0)
    logger.info(f"Label distribution - Judi: {judi_count}, Bukan Judi: {bukan_judi_count}")

    # Probability thresholds for class 1 (gambling) classification
    threshold_low = 0.4
    threshold_high = 0.7

    positive_rows = []  # Predicted as not gambling
    negative_rows = []  # Predicted as gambling
    unknown_rows = []   # Prediction uncertain

    all_predictions = []
    all_probabilities = []
    all_labels = data['label'].tolist()

    max_text_len = 80

    for index, row in data.iterrows():
        text = str(row['text']) if row['text'] is not None else ""
        label = row['label']

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            prob_0 = probs[0][0].item()
            prob_1 = probs[0][1].item()

        # Determine prediction based on probability thresholds
        if prob_1 >= threshold_high:
            prediction = 1  # gambling
        elif prob_1 <= threshold_low:
            prediction = 0  # not gambling
        else:
            prediction = -1  # unknown / uncertain

        if prediction == 1:
            negative_rows.append(row.to_dict())
        elif prediction == 0:
            positive_rows.append(row.to_dict())
        else:
            unknown_rows.append(row.to_dict())

        all_predictions.append(prediction)
        all_probabilities.append(prob_1)

        pred_text = (
            f'judi ({prob_1:.2f})' if prediction == 1 else
            f'bukan judi ({prob_0:.2f})' if prediction == 0 else
            f'unknown ({prob_1:.2f})'
        )
        label_text = 'judi' if label == 1 else 'bukan judi'
        if prediction == -1:
            correctness = 'Ragu-ragu'
        else:
            correctness = 'Betul' if prediction == label else 'Salah prediksi'

        display_text = (text[:max_text_len] + '...') if len(text) > max_text_len else text

        if log_each_row:
            logger.info(f"[Row {index}] Text: \"{display_text}\" | Prediksi: {prediction} ({pred_text}), Label: {label} ({label_text}), Hasil: {correctness}")
            logger.info(f"+========================+")

    os.makedirs("csv/output", exist_ok=True)

    # This marked as new negative because its label is 0 but predicted as 1
    filtered_negative_rows = [row for row in negative_rows if row['label'] == 0]
    pd.DataFrame(filtered_negative_rows).to_csv("csv/output/new_negative.csv", index=False)

    pd.DataFrame(positive_rows).to_csv("csv/output/positive.csv", index=False)
    pd.DataFrame(negative_rows).to_csv("csv/output/negative.csv", index=False)
    pd.DataFrame(unknown_rows).to_csv("csv/output/unknown.csv", index=False)

    logger.info(f"Samples predicted as non gambling saved to csv/output/positive.csv (Count: {len(positive_rows)})")
    logger.info(f"Samples predicted as gambling saved to csv/output/negative.csv (Count: {len(negative_rows)})")
    logger.info(f"Samples predicted as unknown saved to csv/output/unknown.csv (Count: {len(unknown_rows)})")

    # Calculate accuracy, excluding unknown predictions (-1)
    valid_preds_labels = [(p, l) for p, l in zip(all_predictions, all_labels) if p != -1]
    correct = sum([1 for p, l in valid_preds_labels if p == l])
    total = len(valid_preds_labels)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy (excluding unknown): {accuracy:.4f} ({correct}/{total} correct predictions)")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
