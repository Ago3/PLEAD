import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
import pandas as pd
import numpy as np

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
from paead_utils import infer_intent_from_slots


class LitBART(pl.LightningModule):
	# Instantiate the model
	def __init__(self, corpus, seed):
		super().__init__()
		self.tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-{LB_SIZE}', add_prefix_space=True)
		self.tokenizer.add_tokens(ONTOLOGY, special_tokens=True)
		self.model = BartForConditionalGeneration.from_pretrained(f'facebook/bart-{LB_SIZE}')
		self.model.resize_token_embeddings(len(self.tokenizer))
		self.learning_rate = LB_LR
		self.corpus = corpus  # Needed for validation
		self.seed = seed

	# Do a forward pass through the model
	def forward(self, input_ids, **kwargs):
		return self.model(input_ids, **kwargs)
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
		return optimizer

	def training_step(self, batch, batch_idx):
		src_ids, src_mask = self.encode_sentences(batch['text'])
		tgt_ids, _ = self.encode_sentences(batch['label'])

		# Shift the decoder tokens right (but NOT the tgt_ids)
		decoder_input_ids = self.shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

		# Run the model and get the logits
		outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
		lm_logits = outputs[0]  # [batch, seq_length, vocab_size]
		# Create the loss function
		ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
		# Calculate the loss on the un-shifted tokens
		loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

		return {'loss': loss}

	def validation_step(self, batch, batch_idx):
		src_ids, src_mask = self.encode_sentences(batch['text'])
		
		predictions = self.generate_text(src_ids, src_mask)
		predictions_ids = batch['fullID']

		return predictions_ids, predictions

	def validation_epoch_end(self, outs):
		predictions_ids, predictions = list(zip(*outs))
		predictions_ids = list(itertools.chain(*predictions_ids))
		predictions = list(itertools.chain(*predictions))

		with open(LB_PREDICTIONS_FILE, 'w+') as outfile:
			outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}" for pred_id, prediction in zip (predictions_ids, predictions)]))

		
		taskEval = IntentSlotEval(self.corpus, predictions_ids, predictions)
		prod_f1 = taskEval.eval(tests=['f1_cfm_per_token_productions'], arg=False)[0]

		metrics = {'val_prod_f1': prod_f1}
		
		self.log_dict(metrics)

		return metrics

	def test_step(self, batch, batch_idx, dataloader_idx=0):
		src_ids, src_mask = self.encode_sentences(batch['text'])
		
		predictions = self.generate_text(src_ids, src_mask)
		predictions_ids = batch['fullID']

		return predictions_ids, predictions

	def test_epoch_end(self, outs):
		if not self.corpus.task_name == 'aaa':
			outs = [outs]
			self.predictions_file = [LB_PREDICTIONS_FILE]
		else:
			self.predictions_file = LB_AAA_ANSWER_FILES
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

				with open(LB_PREDICTIONS_FILE, 'w+') as outfile:
					outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}" for pred_id, prediction in zip (predictions_ids, predictions)]))

				taskEval = IntentSlotEval(self.corpus, predictions_ids, predictions)
				scores = taskEval.eval(tests=None, arg=False, verbose=False)
				hate_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='hate')
				nothate_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='nothate')
				for subfix, sub_scores in zip(['', '.hate', '.nothate'], [scores, hate_scores, nothate_scores]):
					with open(LB_RES_FILE + subfix, 'a+') as out:
						print(self.seed, LB_RES_FILE + subfix)
						sub_scores = [str(s) for s in sub_scores]
						out.write(f'{LB_EXPERIMENT_NAME}\t{self.seed}\t' + '\t'.join(sub_scores) + '\n')

				metrics = {k: s for k, s in zip(taskEval.available_tests, scores)}

				self.log_dict(metrics)
				return metrics
			
	# Method that generates text using the BartForConditionalGeneration's generate() method
	def generate_text(self, src_ids, attention_mask, eval_beams=LB_BEAM, early_stopping=True, max_len=LB_TGT_MAX_LENGTH):
		''' Function to generate text '''
		generated_ids = self.model.generate(
				src_ids,
				attention_mask=attention_mask,
				use_cache=True,
				decoder_start_token_id = self.tokenizer.pad_token_id,
				num_beams= eval_beams,
				max_length = max_len,
				early_stopping = early_stopping
		)
		decoded_words = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
		final_predictions = []
		for sentence in decoded_words:
			sentence = sentence.replace(',', ' , ')
			pred = sentence.split()
			slot_list = [w for w in pred if w.startswith('SL:')]
			intent = infer_intent_from_slots(slot_list)
			pred = ['[', intent, ','] + pred + [']']
			final_predictions.append(pred)
		return final_predictions

	def encode_sentences(self, sentences, max_length=LB_MAX_LENGTH, pad_to_max_length=False, return_tensors="pt"):
		''' Function that tokenizes a sentence 
			Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
		 	Returns: Dictionary with keys: input_ids, attention_mask, target_ids
		'''

		encoded_dict = self.tokenizer(
				sentences,
				max_length=max_length,
				padding="max_length" if pad_to_max_length else 'longest',
				truncation=True,
				return_tensors=return_tensors,
				add_prefix_space = True
			)

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		return encoded_dict['input_ids'].to(device), encoded_dict['attention_mask'].to(device)


	def shift_tokens_right(self, input_ids, pad_token_id):
		""" Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
		  This is taken directly from modeling_bart.py
		"""
		prev_output_tokens = input_ids.clone()
		index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
		prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
		prev_output_tokens[:, 1:] = input_ids[:, :-1]
		return prev_output_tokens