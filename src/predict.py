import torch
import pickle
from transformers import AutoTokenizer
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    logger.info("Tokenizer loaded successfully.")

    logger.info("Loading model from pickle file...")

    # Override torch.load to map to CPU to fix CUDA deserialization error on CPU-only machine
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
    data = pd.read_csv("csv/predicts.csv")
    logger.info(f"Data loaded successfully with {len(data)} rows.")

    for index, row in data.iterrows():
        text = row['text']
        logger.info(f"Processing row {index}: {text}")

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        logger.info(f"Prediction for row {index}: {prediction} (0: Bukan Judi, 1: Judi)")
        logger.info(f"Komentar ini terindikasi { 'judi' if prediction == 1 else 'bukan judi' }")
        logger.info(f"+========================+")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
