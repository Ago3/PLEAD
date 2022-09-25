import torch
import pandas as pd
from collections import Counter, defaultdict
import string
from paead_info import *
from paead_utils import *
import numpy as np
import itertools
from copy import deepcopy


class TokenTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, args, corpus, split_idxs, is_training_set):
        self.args = args
        self.corpus = corpus
        self.split_idxs = split_idxs
        self.is_training_set = is_training_set
        # Tokenize all senteces (only source, add SOS and EOS)
        self.tokens = []
        self.labels = []
        self.full_ids = []
        self.num_classes = TT_NUM_CLASSES
        for instance in self.corpus.instances:
            if corpus.task_name == 'aaa':
                tokens = instance.tokenized_text
            else:
                tokens, labels = self.get_tokens_to_labels(instance)
                classes_offset = get_slots_offset()
                labels = labels[:, classes_offset:classes_offset + self.num_classes]  # Remove intents and maybe attributes
                self.labels.append(labels)
            self.tokens.append(tokens)
            self.full_ids.append(instance.fullID)
        # Build vocabulary
        self.words = self.find_words()
        self.uniq_words = self.get_uniq_words()

        # define instance using word_idxs
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.pad_token = self.word_to_index[PAD_TOKEN]

        self.words_indexes = [torch.tensor([self.word_to_index[w] for w in instance]) for instance in self.tokens]

        assert TT_OUTPUT_NODES in ['powerset', 'ovo'], f'{TT_OUTPUT_NODES} is not a legal option, choose powerset or ovo.'
        if TT_OUTPUT_NODES == 'ovo':
            self.create_OvO_matrix()
        elif TT_OUTPUT_NODES == 'powerset':
            self.create_OvO_powerset_matrix()

    def get_tokens_to_labels(self, instance):
        tokens = instance.tokenized_text
        labels = [torch.zeros(len(ONTOLOGY)) for i in range(len(tokens))]
        text = instance.text

        def add_slot(span, full_span, label, start_idx):
            if start_idx < 0:
                assert '#' in span, f'The span doesn\'t contain #, but start_idx < 0 ({span})'
                spans = full_span.split('#')
                starts = [text.index(span) for span in spans]
            else:
                spans = [span]
                starts = [start_idx]
            for span, s_idx in zip(spans, starts):
                idx = len(tokenize(text[:s_idx]))
                tokenized_span = span if isinstance(span, list) else tokenize(span)
                for token in tokenized_span:
                    try:
                        labels[idx][ONTOLOGY.index(label)] = 1
                        idx += 1
                    except:
                        print(spans, span, starts, text)
                        exit()

        for field_name, field in instance.subfields.items():
            if not isinstance(field, list):
                field = [field]
            for ff in field:
                add_slot(ff.field_tokenized_text, ff.field_text, f'SL:{title(ff.field_name)}', ff.start_idx)
        return tokens, torch.stack(labels)

    def find_words(self):
        words = [w for instance in self.tokens for w in instance]
        return words + [PAD_TOKEN]

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def create_OvO_matrix(self):
        num_pairs = int((self.num_classes * (self.num_classes - 1)) / 2)
        self.OvO = torch.zeros([self.num_classes, num_pairs])
        change_row = [self.num_classes - 1]
        row = 0
        for column in range(num_pairs):
            if change_row[-1] == column:
                change_row.append(self.num_classes - len(change_row) - 1 + change_row[-1])
                row += 1
            self.OvO[row, column] = 1
        last_column = -1
        for j in range(num_pairs):
            start_row = j+1
            while start_row < self.num_classes:
                last_column += 1
                self.OvO[start_row, last_column] = -1
                start_row += 1

    def create_OvO_powerset_matrix(self):
        self.slots2classes = defaultdict(list)
        self.classes2slots = {}
        c = 0
        slots = ONTOLOGY[get_slots_offset():]
        assert len(slots) == self.num_classes, f'Number of slots in ontology ({len(slots)}) is not equal to the number of classes ({self.num_classes})'
        mtx = get_cooccurence_matrix()
        self.slots2classes[''].append(c)
        self.classes2slots[c] = []
        combs = [[]]
        for i in range(len(slots)):
            new_combs = []
            for comb in combs:
                if not comb or all([mtx[j][i] for j in comb]):
                    new_comb = deepcopy(comb)
                    new_comb.append(i)
                    sorted(new_comb)
                    c += 1
                    self.slots2classes['_'.join([str(s) for s in new_comb])].append(c)
                    self.classes2slots[c] = new_comb
                    new_combs.append(new_comb)
            combs += new_combs

    def __len__(self):
        return len(self.split_idxs)

    def __getitem__(self, index):
        instance = self.corpus.fullids_to_instances[self.split_idxs[index]]
        instances_idx = self.full_ids.index(self.split_idxs[index])
        spans = self.words_indexes[instances_idx]
        sample = {}
        sample['fullID'] = instance.fullID
        sample['text'] = self.words_indexes[instances_idx]
        sample['lengths'] = torch.tensor(sample['text'].size(0))
        if self.labels:
            if TT_OUTPUT_NODES == 'ovo':
                sample['label'] = torch.matmul(self.labels[instances_idx], self.OvO)
            elif TT_OUTPUT_NODES == 'powerset':
                sample['label'] = torch.zeros([sample['text'].size(0), len(self.classes2slots)])
                rows, columns = self.labels[instances_idx].nonzero(as_tuple=True)
                for r in rows:
                    slots = list(self.labels[instances_idx][r, :].nonzero(as_tuple=False))
                    key = '_'.join([str(s.item()) for s in slots])
                    sample['label'][r, self.slots2classes[key]] = 1
        else:
            sample['label'] = torch.tensor([])
        return sample
