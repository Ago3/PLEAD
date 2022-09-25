import torch
from torch import nn, optim
from paead_info import *
from paead_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from .predict import generic_predict
from paead_evaluation import ClassificationEval


def train(dataset, val_dataset, model, args, seed=None):
    model.train()
    model.model.train()

    if RM_OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=RM_LEARNING_RATE)
    print_loss_total = 0

    model, current_epoch, best_valid_score = load_checkpoint(model, RM_MODEL_DIR, RM_MODEL_FILE, seed)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(dataset, batch_size=RM_BATCH_SIZE, pin_memory=True, shuffle=True)

    for i, epoch in enumerate(range(current_epoch + 1, args.max_epochs + 1)):
        model.train()
        model.model.train()
        # model.validation = False
        for batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            print_loss_total += loss.item() / batch['label'].size(0)
        if epoch % RM_PRINT_EVERY == 0 and i > 0:
            print(f'Epoch {epoch} - Average loss on training set: ', print_loss_total / RM_PRINT_EVERY)
            print(f'Epoch {epoch} - Best score on validation set: ', best_valid_score)
            print_loss_total = 0
        valid_score = validate(val_dataset, model, epoch, verbose=True)
        if valid_score > best_valid_score or best_valid_score < 0:
            best_valid_score = valid_score
            save_checkpoint(model, RM_MODEL_DIR, RM_MODEL_FILE, epoch, valid_score, seed)
            print(f'\nBEST SCORE: {best_valid_score} at epoch {epoch}\n')


def validate(dataset, model, epoch, verbose=False):
    model.eval()
    model.model.eval()
    model.validation = True
    predictions, all_predictions_idxs = generic_predict(dataset, model)
    is_binary = 'binary' in dataset.corpus.task_name
    taskEval = ClassificationEval(dataset.corpus, all_predictions_idxs, predictions)
    scores = taskEval.eval(tests=['f1_micro'], arg=is_binary, verbose=False)[0]
    if verbose:
        print(f'--- Epoch {epoch} - F1 score (micro) on validation set: ', scores)
        if not is_binary:
            binary_scores = taskEval.eval(tests=['f1_micro'], arg=True, verbose=False)[0]
            print(f'--- Epoch {epoch} - [binary] F1 score (micro) on validation set: ', binary_scores)
    return scores
