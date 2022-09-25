import torch
from paead_info import *
from paead_utils import load_checkpoint
from tqdm import tqdm


def predict(dataset, model, seed):
    model, _, _ = load_checkpoint(model, TT_MODEL_DIR, TT_MODEL_FILE, seed)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        model.eval()
        model.encoder.eval()
        model.validation = False
        return generic_predict(dataset, model)


def generic_predict(dataset, model):
    predictions = []
    all_predictions_idxs = []
    with open (TT_PREDICTIONS_FILE, 'w+') as log:
        for batch in dataset:
            decoded_words, predictions_idxs = model(batch)
            try:
                log.write(f"{predictions_idxs}\t{''.join(decoded_words)}\n")
            except Exception:
                print(decoded_words)
            predictions += [decoded_words]
            all_predictions_idxs += [predictions_idxs]
    return predictions, all_predictions_idxs
