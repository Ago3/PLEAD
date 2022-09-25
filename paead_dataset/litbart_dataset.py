from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from paead_info import *
from paead_utils import *

from itertools import groupby
from collections import defaultdict
import re


class LitBARTDataset(Dataset):
	def __init__(self, corpus, split_idxs):
		self.corpus = corpus
		self.split_idxs = split_idxs

	def __len__(self):
		return len(self.split_idxs)

	def __getitem__(self, index):
		instance = self.corpus.fullids_to_instances[self.split_idxs[index]]
		sample = {}
		sample['fullID'] = instance.fullID
		sample['text'] = ' '.join(instance.tokenized_text)
		if self.corpus.task_name in ['aaa', 'cad']:
			return sample
		meaning_sketch = [w if w in TOKENS else LB_SKETCH_PLACEHOLDER for w in instance.tokenized_label[3:-1]]
		meaning_sketch = [x[0] for x in groupby(meaning_sketch)]
		sample['sketch'] = ' '.join(meaning_sketch)
		sample['label'] = ' '.join(instance.tokenized_label[3:-1])
		sample['intent'] = RULES.index(instance.rule)
		sample['binary_intent'] = int(instance.rule in HATEFUL_RULES)
		sample['negative_stance'] = 1.0 if 'negative_stance' in instance.subfields else 0.0
		return sample


class LitBARTTaggingDataset(Dataset):
	def __init__(self, corpus, split_idxs):
		# from transformers import BartTokenizerFast
		self.corpus = corpus
		self.split_idxs = split_idxs
		self.__create_OvO_matrix__()

	def __len__(self):
		return len(self.split_idxs)

	def __get_token_labels__(self, instance):
		classes_offset = get_slots_offset()
		text = re.split(r'[^\w]+', instance.text)
		labels = torch.zeros(LMSB_TAG_MAX_LENGTH, len(ONTOLOGY))
		
		tokens = self.tokenizer(' '.join(text))
		sublabels = torch.zeros(LMSB_TAG_MAX_LENGTH, len(ONTOLOGY))
		word2subtoken = defaultdict(list)
		for subtoken, word in enumerate(tokens.word_ids()):
			if word is not None:
				word2subtoken[word].append(subtoken)

		for fieldname, field in instance.subfields.items():
			if not isinstance(field, list): field = [field]
			for f in field:
				l = ONTOLOGY.index(f'SL:{title(f.field_name)}')
				for (s, e) in f.word_indexes:
					labels[s:e, l] = 1
				for i in range(s, e):
					for idx in word2subtoken[i]:
						if idx >= LMSB_TAG_MAX_LENGTH:
							break
						sublabels[idx, l] = 1
		labels = labels[:, classes_offset:classes_offset + LBMS_TAG_CLASSES]  # Remove intents and maybe attributes
		sublabels = sublabels[:, classes_offset:classes_offset + LBMS_TAG_CLASSES]

		labels = torch.matmul(labels, self.OvO)
		sublabels = torch.matmul(sublabels, self.OvO)

		return labels, sublabels, min(len(tokens['input_ids']), LMSB_TAG_MAX_LENGTH)


	def __getitem__(self, index):
		instance = self.corpus.fullids_to_instances[self.split_idxs[index]]
		sample = {}
		sample['fullID'] = instance.fullID
		sample['text'] = ' '.join(re.split(r'[^\w]+', instance.text))
		# Create a vector of slot-labels for each token
		if self.corpus.task_name in ['aaa', 'cad']:
			text = re.split(r'[^\w]+', instance.text)		
			tokens = self.tokenizer(' '.join(text))
			sample['seqlen'] = min(len(tokens['input_ids']), LMSB_TAG_MAX_LENGTH)
			return sample
		sample['token_tags'], sample['subtoken_tags'], sample['seqlen'] = self.__get_token_labels__(instance)
		meaning_sketch = [w if w in TOKENS else LB_SKETCH_PLACEHOLDER for w in instance.tokenized_label[3:-1]]
		meaning_sketch = [x[0] for x in groupby(meaning_sketch)]
		sample['sketch'] = ' '.join(meaning_sketch)
		sample['label'] = ' '.join(instance.tokenized_label[3:-1])
		sample['intent'] = RULES.index(instance.rule)
		sample['binary_intent'] = int(instance.rule in HATEFUL_RULES)
		sample['negative_stance'] = 1.0 if 'negative_stance' in instance.subfields else 0.0
		return sample

	def __create_OvO_matrix__(self):
		num_pairs = int((LBMS_TAG_CLASSES * (LBMS_TAG_CLASSES - 1)) / 2)
		self.OvO = torch.zeros([LBMS_TAG_CLASSES, num_pairs])
		change_row = [LBMS_TAG_CLASSES - 1]
		row = 0
		for column in range(num_pairs):
			if change_row[-1] == column:
				change_row.append(LBMS_TAG_CLASSES - len(change_row) - 1 + change_row[-1])
				row += 1
			self.OvO[row, column] = 1
		last_column = -1
		for j in range(num_pairs):
			start_row = j+1
			while start_row < LBMS_TAG_CLASSES:
				last_column += 1
				self.OvO[start_row, last_column] = -1
				start_row += 1

	def create_OvO_powerset_matrix(self):
		self.slots2classes = defaultdict(list)
		self.classes2slots = {}
		c = 0
		slots = ONTOLOGY[get_slots_offset():]
		assert len(slots) == LBMS_TAG_CLASSES, f'Number of slots in ontology ({len(slots)}) is not equal to the number of classes ({LBMS_TAG_CLASSES})'
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


# Create a dataloading module as per the PyTorch Lightning Docs
class LitBARTDataModule(pl.LightningDataModule):
	def __init__(self, corpus, split_ids, hate_pretraining=0):
		super().__init__()
		self.batch_size = LB_BATCH_SIZE
		self.datasets = [LitBARTDataset(corpus, split_idxs) for split_idxs in split_ids]
		self.hate_pretraining = hate_pretraining
		self.hate_train_dataset = LitBARTDataset(corpus, corpus.only_hate_train_ids) if hate_pretraining else None

	# Load the training, validation and test sets in Pytorch Dataset objects
	def train_dataloader(self):
		if self.hate_train_dataset and self.trainer.current_epoch < self.hate_pretraining:
			train_data = DataLoader(self.hate_train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=2)
			return train_data            
		train_data = DataLoader(self.datasets[0], batch_size = self.batch_size, shuffle=True, num_workers=2)
		return train_data

	def val_dataloader(self):
		val_data = DataLoader(self.datasets[1], batch_size = self.batch_size, num_workers=2)        
		return val_data

	def test_dataloader(self):
		test_data = DataLoader(self.datasets[2], batch_size = self.batch_size, num_workers=2)       
		return test_data

	def aaa_test_dataloaders(self):
		test_dataloaders = [DataLoader(dataset, batch_size = self.batch_size, num_workers=6) for dataset in self.datasets]
		return test_dataloaders


class LitBARTTagDataModule(LitBARTDataModule):
	def __init__(self, corpus, split_ids, hate_pretraining=0):
		super().__init__(corpus, split_ids, hate_pretraining=0)
		self.batch_size = BERTTAG_BATCH_SIZE
		self.datasets = [LitBARTTaggingDataset(corpus, split_idxs) for split_idxs in split_ids]
		self.hate_train_dataset = LitBARTTaggingDataset(corpus, corpus.only_hate_train_ids) if hate_pretraining else None
