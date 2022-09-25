import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput
import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torch

import math
import random
import re
import argparse
import itertools
from copy import deepcopy
from collections import defaultdict

from paead_info import *
from paead_evaluation import IntentSlotEval
from paead_utils import infer_intent_from_slots
from paead_models import LitMSBART, SlotsToIntentModel


class LitMSBARTwithSlot2Intent(LitMSBART):
	# Instantiate the model
	def __init__(self, corpus, predictions_file, res_file, experiment_name, seed):
		super().__init__(corpus, predictions_file, res_file, experiment_name, seed)
		# Create matrix of slot tokens relevant for each intent
		# Relevance can be both positive or negative
		# Only positive stance doesn't affect any intent choice
		self.sketch_token_ids = self.tokenizer.encode(SLOTS, add_special_tokens=False)
		self.slot2intent = SlotsToIntentModel.from_pretrained(LMSBwS2I_S2I_path)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.slot2intent.to(device)
		for p in self.slot2intent.parameters():
			p.requires_grad = False
		self.intent_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
		self.binary_intent_loss_fct = nn.BCELoss()

	def training_step(self, batch, batch_idx):
		src_ids, src_mask = self.encode_sentences(batch['text'])
		sketch_ids, sketch_mask = self.encode_sentences(batch['sketch'])
		tgt_ids, _ = self.encode_sentences(batch['label'])

		# Shift the decoder tokens right (but NOT the tgt_ids)
		decoder_input_ids = self.shift_tokens_right(sketch_ids, self.tokenizer.pad_token_id)
		sketch_decoder_input_ids = self.shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

		# Run the model and get the logits
		outputs, final_outputs = self(src_ids, sketch_ids, tgt_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, sketch_attention_mask=sketch_mask, sketch_decoder_input_ids=sketch_decoder_input_ids)
		lm_logits = outputs[0]  # [batch, seq_length, vocab_size]
		
		# Use sketch tokens to predict intent
		if LMSBwS2I_AGGR == 'sum':
			slot_tokens_logits = lm_logits[:, :, self.sketch_token_ids].sum(dim=1)  # [batch, n_slot_tokens]
		elif LMSBwS2I_AGGR == 'max':
			slot_tokens_logits = lm_logits[:, :, self.sketch_token_ids].max(dim=1).values  # [batch, n_slot_tokens]
		
		normalised_slot_tokens_logits = (slot_tokens_logits - slot_tokens_logits.min(dim=1, keepdim=True).values) / (slot_tokens_logits.max(dim=1, keepdim=True).values - slot_tokens_logits.min(dim=1, keepdim=True).values)
		if LMSBwS2I_NORM:
			intent_scores = self.slot2intent(normalised_slot_tokens_logits)   # [batch, n_intents]
		else:
			intent_scores = self.slot2intent(slot_tokens_logits)   # [batch, n_intents]

		softmax = nn.Softmax(dim=1)
		intent_logits = softmax(intent_scores)
		binary_intent_logits = intent_logits[:, :-1].sum(dim=1)  # [batch]
		
		final_lm_logits = final_outputs[0]  # [batch, seq_length, vocab_size]
		# Create the loss function
		ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')
		# Calculate the loss on the un-shifted tokens
		loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), sketch_ids.view(-1))
		final_loss = ce_loss_fct(final_lm_logits.view(-1, final_lm_logits.shape[-1]), tgt_ids.view(-1))
		# Calculate the loss on the intents
		intent_loss = self.intent_loss_fct(intent_logits, batch['intent'])
		binary_intent_loss = self.binary_intent_loss_fct(binary_intent_logits, batch['binary_intent'].float())
		# Calculate the loss on the stance
		negative_stance_loss_fct = nn.BCELoss(weight=10*batch['negative_stance'])  # BCE has per-instance weights
		stance_loss = negative_stance_loss_fct(normalised_slot_tokens_logits[:, -1], batch['negative_stance'].float())

		sample_weight = ~(batch['binary_intent'].bool()) * LMSBwS2I_NH_WEIGHT
		sample_weight[batch['binary_intent'].bool()] = 1.0
		sample_weight = sample_weight.unsqueeze(1)

		loss = loss.view(lm_logits.shape[:-1])  # [batch, seq_length]
		loss = sample_weight * loss
		loss = loss.mean()
		final_loss = final_loss.view(final_lm_logits.shape[:-1])  # [batch, seq_length]
		final_loss = sample_weight * final_loss
		final_loss = final_loss.mean()

		losses = {'loss': (LMSBwS2I_SKETCH_LOSS_WEIGHT * loss) + (LMSBwS2I_TREE_LOSS_WEIGHT * final_loss) + (LMSBwS2I_INTENT_LOSS_WEIGHT * intent_loss) + (LMSBwS2I_BINARY_INTENT_LOSS_WEIGHT * binary_intent_loss) + (LMSBwS2I_NEGATIVE_STANCE_LOSS_WEIGHT * stance_loss), 'sketch_loss': loss, 'tree_loss': final_loss, 'intent_loss': intent_loss, 'binary_intent_loss': binary_intent_loss, 'negative_stance_loss': stance_loss}

		return losses

	def training_epoch_end(self, losses):
		aggregated_losses = defaultdict(int)
		for d in losses:
			for k, v in d.items():
				aggregated_losses[k] += v
		aggregated_losses = {k: v / len(losses) for k, v in aggregated_losses.items()}
		print('\n', aggregated_losses, '\n')

		self.log_dict(aggregated_losses)
