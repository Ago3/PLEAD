import torch
from torch import nn, optim
from paead_info import *
from paead_utils import *
from tqdm import tqdm
from .predict import generic_predict
from paead_evaluation import IntentSlotEval


def train(dataset, val_dataset, model, args, seed):
    model.train()
    model.encoder.train()

    if TT_OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=TT_LEARNING_RATE)
    elif TT_OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=TT_LEARNING_RATE)
    else:
        assert False, f'Optimizer {TT_OPTIMIZER} not supported for token tagging model.'
    print_loss_total = 0

    model, current_epoch, best_valid_score = load_checkpoint(model, TT_MODEL_DIR, TT_MODEL_FILE, seed)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for i, epoch in enumerate(range(current_epoch + 1, args.max_epochs + 1)):
        model.train()
        model.encoder.train()
        model.validation = False
        for batch in dataset:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            print_loss_total += loss.item() / (batch['label'].size(0) * batch['label'].size(1))
        if epoch % TT_PRINT_EVERY == 0 and i > 0:
            print(f'Epoch {epoch} - Average loss on training set: ', print_loss_total / TT_PRINT_EVERY)
            print(f'Epoch {epoch} - Best score on validation set: ', best_valid_score)
            print_loss_total = 0
        valid_score = validate(val_dataset, model, epoch, verbose=False)
        if valid_score > best_valid_score or best_valid_score < 0:
            best_valid_score = valid_score
            save_checkpoint(model, TT_MODEL_DIR, TT_MODEL_FILE, epoch, valid_score, seed)
            print(f'\nBEST SCORE: {best_valid_score} at epoch {epoch}\n')


def validate(dataset, model, epoch, verbose=False):
    model.eval()
    model.encoder.eval()
    model.validation = True
    predictions, all_predictions_idxs = generic_predict(dataset, model)
    taskEval = IntentSlotEval(dataset.corpus, all_predictions_idxs, predictions)
    scores = taskEval.eval(tests=['f1_cfm_per_token_productions'], arg=False, verbose=False)[0]
    if verbose:
        print(f'--- Epoch {epoch} - Per-token productions F1 on validation set: ', scores)
    return scores
