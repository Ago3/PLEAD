import torch
from paead_info import *
from paead_utils import *
import random


class RobertaDataset(torch.utils.data.Dataset):
    def __init__(self, args, corpus, split_idxs, is_training_set):
        self.args = args
        self.corpus = corpus
        self.split_idxs = split_idxs
        self.is_training_set = is_training_set

    def __len__(self):
        return len(self.split_idxs)

    def __getitem__(self, index):
        sample = {}
        instance = self.corpus.fullids_to_instances[self.split_idxs[index]]
        sample['fullID'] = instance.fullID
        sample['text'] = instance.text
        sample['label'] = instance.label
        return sample
