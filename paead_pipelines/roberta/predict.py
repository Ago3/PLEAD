import torch
from torch.utils.data import DataLoader
from paead_info import *
from paead_utils import load_checkpoint
from tqdm import tqdm
from collections import defaultdict
import random


def predict(dataset, model, zeroshot=None, seed=None):
    model, _, s = load_checkpoint(model, RM_MODEL_DIR, RM_MODEL_FILE, seed)
    print('Best score: ', s)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        model.eval()
        model.model.eval()
        # model.validation = False
        return generic_predict(dataset, model)


def generic_predict(dataset, model):
    answers = defaultdict(list)
    ids = []
    all_predictions = []
    all_ids = []
    loader = DataLoader(dataset, batch_size=RM_BATCH_SIZE, pin_memory=True)
    for batch in tqdm(loader):
        predictions, predictions_idxs = model(batch)
        if RM_USE_POLICY:
            for fullid, ans in zip(predictions_idxs, predictions):
                idx = '_'.join(fullid.split('_')[:-1])
                policy_idx = int(fullid.split('_')[-1])
                if ans:
                    answers[idx].append(policy_idx)
                ids.append(idx)
        else:
            ids += predictions_idxs
            all_predictions += [p.item() for p in predictions.cpu()]
    if RM_USE_POLICY:
        for k in ids:
            if k not in all_ids:
                v = answers[k]
                if not v:
                    all_predictions.append(RULES.index('nothate'))
                else:
                    rule = random.choice(v)
                    all_predictions.append(rule)
                all_ids.append(k)
        ids = all_ids
    return all_predictions, ids
