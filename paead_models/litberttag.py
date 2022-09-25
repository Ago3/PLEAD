import transformers
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BertTokenizerFast
import pandas as pd
import numpy as np
from torch import nn

import torch.nn.functional as F
import pytorch_lightning as pl
import torch

import math
import random
import re
import argparse
import itertools

from paead_info import *
from paead_evaluation import IntentSlotEval
from paead_utils import *


class LitBERTTag(pl.LightningModule):
	# Instantiate the model
	def __init__(self, corpus, seed, OvO):
		super().__init__()
		self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		self.corpus = corpus
		config = BertConfig.from_pretrained('bert-base-uncased')
		self.output_nodes = int((BERTTAG_CLASSES * (BERTTAG_CLASSES - 1)) / 2)
		config.num_labels = self.output_nodes
		self.model = BertModel(config)
		self.learning_rate = BERTTAG_LR
		self.corpus = corpus  # Needed for validation
		self.seed = seed
		self.OvO = OvO
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=2)
		self.sigmoid = nn.Sigmoid()
		hidden_size = 768
		self.fc1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
		self.fc2 = nn.Linear(hidden_size, self.output_nodes, bias=False)
		self.fc3 = nn.Linear(self.output_nodes, self.output_nodes, bias=False)
		self.dropout = nn.Dropout(p=BERTTAG_DROPOUT)
		self.relu = nn.ReLU()

	# Do a forward pass through the model
	def forward(self, batch, **kwargs):
		inputs = self.encode_sentences(batch['text'])
		inputs.to(self.device)
		output = self.model(**inputs)
		att_mask = inputs['attention_mask']
		att_mask[:, 0] = 0  # ignore special characters
		att_mask[:, batch['seqlen'] - 1] = 0  # ignore special characters
		hidden_states = output.last_hidden_state
		hidden_states = torch.cat([hidden_states, (hidden_states[:, 0, :].unsqueeze(1).repeat(1, hidden_states.size(1), 1))], dim=2)
		hidden_states[att_mask==0] = 0.0
		logits = self.fc2(self.relu(self.dropout(self.fc1(hidden_states))))
		logits = self.fc3(self.relu(logits).unsqueeze(0)).squeeze(0)  # connect the probabilities of all the slots together
		att_mask = att_mask.unsqueeze(2).repeat(1, 1, self.output_nodes)
		return logits, inputs['input_ids'], att_mask
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
		return optimizer

	def training_step(self, batch, batch_idx):
		# Run the model and get the logits
		output, input_ids, att_mask = self(batch)
		# Create the loss function
		probs = ((self.tanh(output) + 1) / 2).clamp(0.00001, 0.99999)
		labels = ((batch['subtoken_tags'] + 1.0) / 2.0)
		labels = labels[:, :probs.size(1), :]
		
		ce_loss_fct = torch.nn.BCELoss(reduction='none')
		
		loss = ce_loss_fct(probs, labels)
		loss = loss.masked_select(att_mask.bool()).mean()

		return {'loss': loss}

	def validation_step(self, batch, batch_idx):
		output, input_ids, att_mask = self(batch)
		output.to(self.device)
		self.OvO = self.OvO.to(self.device)
		probs = self.tanh(output)
		probs[att_mask==0] = 0.0
		probs = torch.matmul(probs, self.OvO.transpose(0, 1))
		preds = (probs > BERTTAG_THRESHOLD)
		
		predictions = self.generate_text(input_ids, preds, batch['seqlen'])
		predictions_ids = batch['fullID']

		return predictions_ids, predictions

	def validation_epoch_end(self, outs):
		predictions_ids, predictions = list(zip(*outs))
		predictions_ids = list(itertools.chain(*predictions_ids))
		predictions = list(itertools.chain(*predictions))

		with open(BERTTAG_PREDICTIONS_FILE, 'w+') as outfile:
			outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}" for pred_id, prediction in zip (predictions_ids, predictions)]))

		
		taskEval = IntentSlotEval(self.corpus, predictions_ids, predictions)
		prod_f1 = taskEval.eval(tests=['f1_cfm_per_token_productions'], arg=False)[0]

		metrics = {'val_prod_f1': prod_f1}
		
		self.log_dict(metrics)

		return metrics

	def test_step(self, batch, batch_idx, dataloader_idx=0):
		output, input_ids, att_mask = self(batch)
		output.to(self.device)
		self.OvO = self.OvO.to(self.device)
		# probs = self.softmax(output.logits)
		probs = self.tanh(output)
		probs[att_mask==0] = 0.0
		probs = torch.matmul(probs, self.OvO.transpose(0, 1))
		# probs -= probs.min(2, keepdim=True)[0]
		# probs /= probs.max(2, keepdim=True)[0]
		preds = (probs > BERTTAG_THRESHOLD)
		
		predictions = self.generate_text(input_ids, preds, batch['seqlen'])
		predictions_ids = batch['fullID']

		return predictions_ids, predictions

	def test_epoch_end(self, outs):
		if not self.corpus.task_name == 'aaa':
			outs = [outs]
			self.predictions_file = [BERTTAG_PREDICTIONS_FILE]
		else:
			self.predictions_file = BERTTAG_AAA_ANSWER_FILES
		for dataloader_outs, predictions_file in zip(outs, self.predictions_file):
			predictions_ids, predictions = list(zip(*dataloader_outs))
			predictions_ids = list(itertools.chain(*predictions_ids))
			predictions = list(itertools.chain(*predictions))

			if self.corpus.task_name == 'aaa':

				with open(predictions_file + f'.exp.{self.seed}', 'w+') as outfile:
					outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}" for pred_id, prediction in zip (predictions_ids, predictions)]))

				binary_predictions = [0 if 'IN:NotHateful' in prediction else 1 for prediction in predictions]
				instances = [self.corpus.fullids_to_instances[pred_id] for pred_id in predictions_ids]

				with open(predictions_file + f'.{self.seed}', 'w+') as outfile:
					outfile.write('\n'.join([f"{instance.text}\t{int(instance.rule == 'hate')}\t{prediction}" for instance, prediction in zip (instances, binary_predictions)]))
			
			else:

				with open(BERTTAG_PREDICTIONS_FILE, 'w+') as outfile:
					outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}" for pred_id, prediction in zip (predictions_ids, predictions)]))

				taskEval = IntentSlotEval(self.corpus, predictions_ids, predictions)
				scores = taskEval.eval(tests=None, arg=False, verbose=False)
				hate_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='hate')
				nothate_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='nothate')

				derogation_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: 'derogation_span' in x.subfields))
				animosity_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: 'animosity_span' in x.subfields))
				for subfix, sub_scores in zip(['', '.hate', '.nothate', '.derogation', '.animosity'], [scores, hate_scores, nothate_scores, derogation_scores, animosity_scores]):
					with open(BERTTAG_RES_FILE + subfix, 'a+') as out:
						print(self.seed, BERTTAG_RES_FILE + subfix)
						sub_scores = [str(s) for s in sub_scores]
						out.write(f'{BERTTAG_EXPERIMENT_NAME}\t{self.seed}\t' + '\t'.join(sub_scores) + '\n')

				metrics = {k: s for k, s in zip(taskEval.available_tests, scores)}

				self.log_dict(metrics)
				return metrics

	def encode_sentences(self, sentences, max_length=BERTTAG_MAX_LENGTH, pad_to_max_length=False, return_tensors="pt"):
		''' Function that tokenizes a sentence 
			Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
		 	Returns: Dictionary with keys: input_ids, attention_mask, target_ids
		'''

		encoded_dict = self.tokenizer(
				sentences,
				max_length=max_length,
				padding="max_length" if pad_to_max_length else 'longest',
				truncation=True,
				add_special_tokens=True,
				return_tensors=return_tensors,
			)

		for k, v in encoded_dict.items():
			v.to(self.device)
		return encoded_dict

	def generate_text(self, batch_tokens, batch_predictions, lengths):
		# I think I can ignore lengths as only looking for True values ?
		linearised_trees = []
		# for each chosen slot label:
		for tokens, predictions in zip(batch_tokens, batch_predictions):
			chosen_labels = torch.any(predictions, dim=0, keepdim=False).nonzero(as_tuple=False)
			label2chunks = dict()
			for label in chosen_labels:
				if not is_attribute(label + get_slots_offset()):
					attributes_idxs = get_attributes(label + get_slots_offset())
					chosen_attributes = torch.tensor(list(set([v.item() for v in chosen_labels]) & set(attributes_idxs)))
					new_slots = dict()
					slot2attrs = defaultdict(list)
					for i, token in enumerate(tokens):
						if predictions[i][label] or (i-1 in new_slots and self.tokenizer.decode(token)[0:2] == '##'):
							#if adjacent tokens have this label, merge
							if i-1 in new_slots:
								new_slots[i] = new_slots[i-1] + [token]
								del new_slots[i-1]
								if chosen_attributes.size(0) > 0 and predictions[i][chosen_attributes].nonzero().size(0) > 0:
									slot2attrs[i] = slot2attrs[i-1] + chosen_attributes[predictions[i][chosen_attributes].nonzero()].squeeze(0).tolist()
								elif slot2attrs[i-1]:
									slot2attrs[i] = slot2attrs[i-1]
									del slot2attrs[i-1]
							elif self.tokenizer.decode(token)[0:2] == '##':
								#If it's a subtoken, add previous parts of the token
								j = i-1
								while self.tokenizer.decode(tokens[j])[0:2] == '##':
									j -= 1
								new_slots[i] = tokens[j:i+1].tolist()
							else:
								new_slots[i] = [token]
								if chosen_attributes.size(0) > 0 and predictions[i][chosen_attributes].nonzero().size(0) > 0:
									attrs = chosen_attributes[predictions[i][chosen_attributes].nonzero()].squeeze(0).tolist()
									slot2attrs[i] += attrs
					if len(new_slots) > 1 and not is_protected_characteristic(label + get_slots_offset()):
						# While you can have multiple PCs, for the other slots just merge adding '#'
						keys = list(new_slots.keys())
						sorted(keys)
						for k in keys[1:]:
							new_slots[keys[0]] += self.tokenizer.encode('#', add_special_tokens=False) + new_slots[k]
							del new_slots[k]
							if k in slot2attrs:
								slot2attrs[keys[0]] += slot2attrs[k]
								del slot2attrs[k]
					for k, v in new_slots.items():
						#if any of the merged slots have an attribute, assign the attribute to the whole slot. TODO: What if multiple attributes?
						v = self.tokenizer.decode(v)
						if k in slot2attrs:
							attr_dict = {slot2attrs[k][0]: v}
							label2chunks[label.item()] = attr_dict
						else:
							label2chunks[label.item()] = v
			# use assigned labels and word ids to generate string
			linearised_tree = ['[', infer_intent_from_slots(list(label2chunks.keys())), ',']
			for k, v in label2chunks.items():
				linearised_tree += ['[', get_ontology_name(k + get_slots_offset()), ',']
				if isinstance(v, dict):
					kk = list(v.keys())[0]
					span = v[kk]
					linearised_tree += ['[', get_ontology_name(kk + get_slots_offset()), ',', span, ']', ',']
				else:
					span = v
					linearised_tree += [span, ']', ',']
			linearised_tree = linearised_tree[:-1] + [']']
			linearised_trees.append(linearised_tree)
		return linearised_trees
