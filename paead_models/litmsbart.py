import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput
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
from copy import deepcopy

from paead_info import *
from paead_evaluation import IntentSlotEval
from paead_utils import infer_intent_from_slots


class MeaningSketchesBart(BartForConditionalGeneration):
	def __init__(self):
		super().__init__(BartConfig())
		self.tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-{LMSB_SIZE}', add_prefix_space=True)
		self.tokenizer.add_tokens(ONTOLOGY, special_tokens=True)
		self.model = BartForConditionalGeneration.from_pretrained(f'facebook/bart-{LMSB_SIZE}')
		self.model.resize_token_embeddings(len(self.tokenizer))
		self.original_decoder = deepcopy(self.model.model.decoder)
		self.sketch_guided_decoder = deepcopy(self.model.model.decoder)
		self.original_lm_head = deepcopy(self.model.lm_head)
		self.sketch_guided_lm_head = deepcopy(self.model.lm_head)

	def forward(self, **kwargs):
		self.original_decoder = deepcopy(self.model.model.decoder)
		self.model.model.decoder = deepcopy(self.sketch_guided_decoder)
		self.original_lm_head = deepcopy(self.model.lm_head)
		self.model.lm_head = deepcopy(self.sketch_guided_lm_head)
		output = self.model(**kwargs)
		self.model.model.decoder = self.original_decoder
		self.model.lm_head = self.original_lm_head
		return output


class LitMSBART(pl.LightningModule):
	# Instantiate the model
	def __init__(self, corpus, predictions_file, res_file, experiment_name, seed):
		super().__init__()
		self.ms_model = MeaningSketchesBart()
		self.tokenizer = self.ms_model.tokenizer
		self.model = self.ms_model.model
		self.learning_rate = LMSB_LR
		self.corpus = corpus  # Needed for validation
		self.predictions_file = predictions_file
		self.res_file = res_file
		self.experiment_name = experiment_name
		self.seed = seed

	# Do a forward pass through the model
	def forward(self, input_ids, sketch_ids, tgt_ids, **kwargs):
		# Encode input text and decode sketch
		output = self.input_encoding_sketch_decoding(input_ids, **kwargs)
		
		# # Create the sketch without beam search, just take the max_prob token
		if LMSB_ENCODE_INPUT_TWICE:
			# Encode gold sketch + input again
			kwargs['sketch_attention_mask'] = torch.cat([kwargs['attention_mask'], kwargs['sketch_attention_mask']], dim=1)
			sketch_encoding = self.sketch_encoding(torch.cat([input_ids, sketch_ids], dim=1), **kwargs)
		else:
			# Encode gold sketch
			sketch_encoding = self.sketch_encoding(sketch_ids, **kwargs)
		
		final_output = self.sketch_guided_output_decoding(sketch_ids, tgt_ids, (output.encoder_last_hidden_state,
output.encoder_hidden_states, output.encoder_attentions), sketch_encoding, **kwargs)
		return output, final_output

	def input_encoding_sketch_decoding(self, input_ids, **kwargs):

		output_attentions = kwargs['output_attentions'] if 'output_attentions' in kwargs else self.model.model.config.output_attentions
		output_hidden_states = (
			kwargs['output_hidden_states'] if 'output_hidden_states' in kwargs else self.model.model.config.output_hidden_states
		)
		use_cache = kwargs['use_cache'] if 'use_cache' in kwargs else self.model.model.config.use_cache
		return_dict = kwargs['return_dict'] if 'return_dict' in kwargs else self.model.model.config.use_return_dict

		encoder_outputs = kwargs['encoder_outputs'] if 'encoder_outputs' in kwargs else None
		attention_mask = kwargs['attention_mask'] if 'attention_mask' in kwargs else None
		head_mask = kwargs['head_mask'] if 'head_mask' in kwargs else None
		inputs_embeds = kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs else None
		decoder_input_ids = kwargs['decoder_input_ids'] if 'decoder_input_ids' in kwargs else None
		decoder_attention_mask = kwargs['decoder_attention_mask'] if 'decoder_attention_mask' in kwargs else None
		decoder_head_mask = kwargs['decoder_head_mask'] if 'decoder_head_mask' in kwargs else None
		cross_attn_head_mask = kwargs['cross_attn_head_mask'] if 'cross_attn_head_mask' in kwargs else None
		past_key_values = kwargs['past_key_values'] if 'past_key_values' in kwargs else None
		decoder_inputs_embeds = kwargs['decoder_inputs_embeds'] if 'decoder_inputs_embeds' in kwargs else None
		labels = kwargs['labels'] if 'labels' in kwargs else None

		if encoder_outputs is None:
			encoder_outputs = self.model.model.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)
		# If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
		elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
			encoder_outputs = BaseModelOutput(
				last_hidden_state=encoder_outputs[0],
				hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
				attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
			)

		# decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs = self.model.model.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=encoder_outputs[0],
			encoder_attention_mask=attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			outputs = decoder_outputs + encoder_outputs
		else:
			outputs = Seq2SeqModelOutput(
				last_hidden_state=decoder_outputs.last_hidden_state,
				past_key_values=decoder_outputs.past_key_values,
				decoder_hidden_states=decoder_outputs.hidden_states,
				decoder_attentions=decoder_outputs.attentions,
				cross_attentions=decoder_outputs.cross_attentions,
				encoder_last_hidden_state=encoder_outputs.last_hidden_state,
				encoder_hidden_states=encoder_outputs.hidden_states,
				encoder_attentions=encoder_outputs.attentions,
			)

		lm_logits = self.model.lm_head(outputs[0]) + self.model.final_logits_bias

		masked_lm_loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size), labels.view(-1))

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		return Seq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=outputs.past_key_values,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
		)

	def sketch_encoding(self, input_ids, **kwargs):
		output_attentions = kwargs['output_attentions'] if 'output_attentions' in kwargs else self.model.model.config.output_attentions
		output_hidden_states = (
			kwargs['output_hidden_states'] if 'output_hidden_states' in kwargs else self.model.model.config.output_hidden_states
		)
		use_cache = kwargs['use_cache'] if 'use_cache' in kwargs else self.model.model.config.use_cache
		return_dict = kwargs['return_dict'] if 'return_dict' in kwargs else self.model.model.config.use_return_dict

		encoder_outputs = kwargs['encoder_outputs'] if 'encoder_outputs' in kwargs else None
		attention_mask = kwargs['sketch_attention_mask'] if 'sketch_attention_mask' in kwargs else None
		head_mask = kwargs['head_mask'] if 'head_mask' in kwargs else None
		inputs_embeds = kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs else None

		if encoder_outputs is None:
			encoder_outputs = self.model.model.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)
		# If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
		elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
			encoder_outputs = BaseModelOutput(
				last_hidden_state=encoder_outputs[0],
				hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
				attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
			)
		return encoder_outputs

	def sketch_guided_output_decoding(self, sketch_ids, tgt_ids, input_encoding, sketch_encoding, **kwargs):
		# Base case: just concat input and sketch encoding
		if LMSB_ENCODE_INPUT_TWICE:
			encoder_outputs = sketch_encoding[0]
			attention_mask = kwargs['sketch_attention_mask']
		else:
			encoder_outputs = torch.cat([input_encoding[0], sketch_encoding[0]], dim=1)
			attention_mask = torch.cat([kwargs['attention_mask'], kwargs['sketch_attention_mask']], dim=1)

		output_attentions = kwargs['output_attentions'] if 'output_attentions' in kwargs else self.model.model.config.output_attentions
		output_hidden_states = (
			kwargs['output_hidden_states'] if 'output_hidden_states' in kwargs else self.model.model.config.output_hidden_states
		)
		use_cache = kwargs['use_cache'] if 'use_cache' in kwargs else self.model.model.config.use_cache
		return_dict = kwargs['return_dict'] if 'return_dict' in kwargs else self.model.model.config.use_return_dict

		decoder_input_ids = kwargs['sketch_decoder_input_ids'] if 'sketch_decoder_input_ids' in kwargs else None
		decoder_attention_mask = kwargs['decoder_attention_mask'] if 'decoder_attention_mask' in kwargs else None
		decoder_head_mask = kwargs['decoder_head_mask'] if 'decoder_head_mask' in kwargs else None
		cross_attn_head_mask = kwargs['cross_attn_head_mask'] if 'cross_attn_head_mask' in kwargs else None
		past_key_values = kwargs['past_key_values'] if 'past_key_values' in kwargs else None
		decoder_inputs_embeds = kwargs['decoder_inputs_embeds'] if 'decoder_inputs_embeds' in kwargs else None
		labels = kwargs['labels'] if 'labels' in kwargs else None

		# decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs = self.ms_model.sketch_guided_decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=encoder_outputs,
			encoder_attention_mask=attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			outputs = decoder_outputs + encoder_outputs
		else:
			outputs = Seq2SeqModelOutput(
				last_hidden_state=decoder_outputs.last_hidden_state,
				past_key_values=decoder_outputs.past_key_values,
				decoder_hidden_states=decoder_outputs.hidden_states,
				decoder_attentions=decoder_outputs.attentions,
				cross_attentions=decoder_outputs.cross_attentions,
				encoder_last_hidden_state=encoder_outputs,
				encoder_hidden_states=None,
				encoder_attentions=None,
			)

		lm_logits = self.ms_model.sketch_guided_lm_head(outputs[0]) + self.model.final_logits_bias

		masked_lm_loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size), labels.view(-1))

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		return Seq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=outputs.past_key_values,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
		)
	
	def configure_optimizers(self):
		# for name, p in self.named_parameters():
		# 	print(name)
		optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
		return optimizer

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
		final_lm_logits = final_outputs[0]  # [batch, seq_length, vocab_size]
		# Create the loss function
		ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
		# Calculate the loss on the un-shifted tokens
		loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), sketch_ids.view(-1))
		final_loss = ce_loss_fct(final_lm_logits.view(-1, final_lm_logits.shape[-1]), tgt_ids.view(-1))

		return {'loss':loss + final_loss}

	def validation_step(self, batch, batch_idx):
		src_ids, src_mask = self.encode_sentences(batch['text'])
		
		predictions, sketches = self.generate_text(src_ids, src_mask)
		predictions_ids = batch['fullID']

		return predictions_ids, predictions, sketches

	def validation_epoch_end(self, outs):
		predictions_ids, predictions, sketches = list(zip(*outs))
		predictions_ids = list(itertools.chain(*predictions_ids))
		predictions = list(itertools.chain(*predictions))
		sketches = list(itertools.chain(*sketches))

		with open(self.predictions_file, 'w+') as outfile:
			outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}\t{sketch}" for pred_id, prediction, sketch in zip (predictions_ids, predictions, sketches)]))

		if self.corpus.task_name in ['aaa', 'cad']:
			return {}
		
		taskEval = IntentSlotEval(self.corpus, predictions_ids, predictions)
		prod_f1 = taskEval.eval(tests=['f1_cfm_per_token_productions'], arg=False)[0]

		metrics = {'val_prod_f1': prod_f1}
		self.log_dict(metrics)

		return metrics

	def test_step(self, batch, batch_idx, dataloader_idx=0):
		src_ids, src_mask = self.encode_sentences(batch['text'])
		
		predictions, sketches = self.generate_text(src_ids, src_mask)
		predictions_ids = batch['fullID']

		return predictions_ids, predictions, sketches

	def test_epoch_end(self, outs):
		if not self.corpus.task_name in ['aaa', 'cad']:
			outs = [outs]
			self.predictions_file = [self.predictions_file]
		# if self.corpus.task_name in ['cad']:
			# outs = [outs]
		for dataloader_outs, predictions_file in zip(outs, self.predictions_file):
			predictions_ids, predictions, sketches = list(zip(*dataloader_outs))
			predictions_ids = list(itertools.chain(*predictions_ids))
			predictions = list(itertools.chain(*predictions))
			sketches = list(itertools.chain(*sketches))

			if self.corpus.task_name in ['aaa', 'cad']:

				with open(predictions_file + f'.exp.{self.seed}', 'w+') as outfile:
					outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}\t{sketch}" for pred_id, prediction, sketch in zip (predictions_ids, predictions, sketches)]))

				binary_predictions = [0 if 'IN:NotHateful' in prediction else 1 for prediction in predictions]
				instances = [self.corpus.fullids_to_instances[pred_id] for pred_id in predictions_ids]

				with open(predictions_file + f'.{self.seed}', 'w+') as outfile:
					outfile.write('\n'.join([f"{instance.text}\t{int(instance.rule == 'hate')}\t{prediction}" for instance, prediction in zip (instances, binary_predictions)]))
			
			else:

				with open(predictions_file, 'w+') as outfile:
					outfile.write('\n'.join([f"{pred_id}\t{' '.join(prediction)}\t{sketch}" for pred_id, prediction, sketch in zip (predictions_ids, predictions, sketches)]))

				taskEval = IntentSlotEval(self.corpus, predictions_ids, predictions)
				scores = taskEval.eval(tests=None, arg=False, verbose=False)
				metrics = {k: s for k, s in zip(taskEval.available_tests, scores)}
				hate_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='hate')
				hate_metrics = {f'hate_{k}': s for k, s in zip(taskEval.available_tests, hate_scores)}
				nothate_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='nothate')
				nothate_metrics = {f'nothate_{k}': s for k, s in zip(taskEval.available_tests, nothate_scores)}
				threatening_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: x.rule == 'threatening'))
				comparison_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: x.rule == 'comparison'))
				hatecrime_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: x.rule == 'hatecrime'))
				derogation_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: 'derogation_span' in x.subfields))
				animosity_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: 'animosity_span' in x.subfields))
				explicit_scores = taskEval.eval(tests=None, arg=False, verbose=False, true_class='all', instancefilter=(lambda x: ('animosity_span' not in x.subfields) and (not x.rule == 'nothate')))

				for subfix, sub_scores in zip(['', '.hate', '.nothate', '.threatening', '.comparison', '.hatecrime', '.derogation', '.animosity', '.explicit'], [scores, hate_scores, nothate_scores, threatening_scores, comparison_scores, hatecrime_scores, derogation_scores, animosity_scores, explicit_scores]):
					with open(self.res_file + subfix, 'a+') as out:
						sub_scores = [str(s) for s in sub_scores]
						out.write(f'{self.experiment_name}\t{self.seed}\t' + '\t'.join(sub_scores) + '\n')
				metrics = {**metrics, **hate_metrics, **nothate_metrics}

				self.log_dict(metrics)
				return metrics
	
	# Method that generates text using the BartForConditionalGeneration's generate() method
	def generate_text(self, src_ids, attention_mask, eval_beams=LMSB_BEAM, early_stopping=True, max_len=LMSB_TGT_MAX_LENGTH):
		''' Function to generate text '''
		generated_sketch_ids = self.model.generate(
				src_ids, #  None
				attention_mask=attention_mask,
				use_cache=True,
				decoder_start_token_id = self.tokenizer.pad_token_id,
				num_beams= eval_beams,
				max_length = max_len,
				early_stopping = early_stopping
		)
		decoded_sketches = self.tokenizer.batch_decode(generated_sketch_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
		sketch_attention_mask = generated_sketch_ids.ne(self.tokenizer.pad_token_id).long()
		joint_src_sketch_ids = torch.cat([src_ids, generated_sketch_ids], dim=1)
		joint_src_sketch_mask = torch.cat([attention_mask, sketch_attention_mask], dim=1)
		generated_ids = self.ms_model.generate(
				joint_src_sketch_ids,
				attention_mask=joint_src_sketch_mask,
				use_cache=True,
				decoder_start_token_id = self.tokenizer.pad_token_id,
				num_beams= eval_beams,
				max_length = max_len,
				early_stopping = early_stopping
		)
		decoded_words = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
		final_predictions = []
		for sentence in decoded_words:
			sentence = re.sub( r'([,.!?;#])', r' \1 ', sentence)
			# sentence = sentence.replace(',', ' , ')
			pred = sentence.split()
			slot_list = [w for w in pred if w.startswith('SL:')]
			intent = infer_intent_from_slots(slot_list)
			pred = ['[', intent, ','] + pred + [']']
			final_predictions.append(pred)
		return final_predictions, decoded_sketches

	def encode_sentences(self, sentences, max_length=LMSB_MAX_LENGTH, pad_to_max_length=False, return_tensors="pt"):
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