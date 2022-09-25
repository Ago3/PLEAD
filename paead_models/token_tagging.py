import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from paead_info import TT_EMBEDDING_SIZE, TT_HIDDEN_SIZE, TT_NUM_LAYERS, TT_NUM_CLASSES, TT_NEGATIVE_SAMPLES, TT_THRESHOLD, TT_DROPOUT
from paead_utils import *
import numpy as np
from collections import defaultdict


class EncoderRNN(nn.Module):
	def __init__(self, embedding_dim, hidden_size, n_vocab, num_layers, device, bidirectional=True):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.device = device
		self.n_directions = 2 if bidirectional else 1
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.num_layers, bidirectional=bidirectional, batch_first=True)  #dropout=0.2,
		self.relu = nn.ReLU()


	def forward(self, x, batch_size=None):
		if batch_size is None:
			batch_size = x.size(0)
		h_0 = Variable(torch.zeros(self.num_layers * self.n_directions, batch_size, self.hidden_size)).to(self.device) #hidden state
		c_0 = Variable(torch.zeros(self.num_layers * self.n_directions, batch_size, self.hidden_size)).to(self.device) #internal state
		# Propagate input through LSTM
		output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
		out = self.relu(output.squeeze())  # [NUM_TOKENS, HIDDEN x N_DIRECTIONS]
		return out

	

class TokenTaggingModel(nn.Module):
	def __init__(self, dataset, embeddings, bidirectional=True, use_negative_sampling=False):
		super(TokenTaggingModel, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.lstm_size = TT_HIDDEN_SIZE
		self.embedding_dim = TT_EMBEDDING_SIZE
		self.num_layers = TT_NUM_LAYERS
		self.num_classes = TT_NUM_CLASSES
		if TT_OUTPUT_NODES == 'ovo':
			self.output_nodes = int((self.num_classes * (self.num_classes - 1)) / 2)
			self.activation = nn.Tanh()
		elif TT_OUTPUT_NODES == 'powerset':
			self.output_nodes = len(dataset.classes2slots)
			self.activation = nn.Softmax()
		self.use_negative_sampling = use_negative_sampling
		n_vocab = len(dataset.uniq_words)
		if embeddings is None:
			self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=TT_EMBEDDING_SIZE, padding_idx=dataset.pad_token).to(self.device)
		else:
			self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), padding_idx=dataset.pad_token).to(self.device)
		self.encoder = EncoderRNN(self.embedding_dim, self.lstm_size, n_vocab, self.num_layers, self.device, bidirectional=bidirectional)
		# if use_negative_sampling:
		# 	pos_weight = torch.ones([TT_NEGATIVE_SAMPLES + 1]) * 10
		# else:
		# 	pos_weight = torch.ones([self.num_classes]) #* 10
		self.criterion = nn.BCELoss()
		self.fc1 = nn.Linear(self.encoder.n_directions *  self.num_layers * self.lstm_size, self.lstm_size, bias=False)
		self.fc2 = nn.Linear(self.lstm_size, self.output_nodes, bias=False)
		self.fc3 = nn.Linear(self.output_nodes, self.output_nodes, bias=False)
		self.dropout = nn.Dropout(p=TT_DROPOUT)
		self.relu = nn.ReLU()
		self.validation = False
		# self.label_probs = dataset.label_probs
		self.pad_token = dataset.pad_token
		self.index_to_word = dataset.index_to_word
		self.word_to_index = dataset.word_to_index
		if TT_OUTPUT_NODES == 'ovo':
			self.OvO = dataset.OvO.to(self.device)
		elif TT_OUTPUT_NODES == 'powerset':
			self.classes2slots = dataset.classes2slots

	def forward(self, x):
		current_batch_size = 1
		x['text'] = x['text'].to(self.device).unsqueeze(0)
		x['label'] = x['label'].to(self.device)
		loss = 0
		embedded = self.embedding(x['text'])
		encoder_output = self.encoder(embedded, current_batch_size)
		
		logits = self.fc2(self.relu(self.dropout(self.fc1(encoder_output))))
		logits = self.fc3(self.relu(logits).unsqueeze(0)).squeeze()  # connect the probabilities of all the slots together
		if len(logits.shape) == 1:
			logits = logits.unsqueeze(0)
		probs = self.activation(logits)
		if self.training and not self.validation:
			if TT_OUTPUT_NODES == 'ovo':
				probs = ((probs + 1) / 2).clamp(0.00001, 0.99999)
				labels = (x['label'] + 1) / 2
			elif TT_OUTPUT_NODES == 'powerset':
				labels = x['label']
			loss = self.criterion(probs, labels)
		if not self.training:
			if TT_OUTPUT_NODES == 'ovo':
				probs = torch.matmul(probs, self.OvO.transpose(0, 1))
				preds = (probs > TT_THRESHOLD)
			elif TT_OUTPUT_NODES == 'powerset':
				preds = torch.argmax(probs, dim=1)
			return self.convert_instance_predictions_into_linearised_tree(x, preds), x['fullID']
		return loss


	def convert_instance_predictions_into_linearised_tree(self, x, preds):
		tokens = x['text'][0, :]
		if TT_OUTPUT_NODES == 'ovo':
			predictions = preds
		elif TT_OUTPUT_NODES == 'powerset':
			predictions = torch.zeros([preds.size(0), self.num_classes], device=self.device)
			for i in range(preds.size(0)):
				chosen_labels = self.classes2slots[preds[i].item()]
				if chosen_labels:
					predictions[i, chosen_labels] = 1
				predictions = predictions.bool()
		# for each chosen slot label:
		chosen_labels = torch.any(predictions, dim=0, keepdim=False).nonzero(as_tuple=False)
		label2chunks = dict()
		# label2chunkids = defaultdict(list)
		for label in chosen_labels:
			if not is_attribute(label + get_slots_offset()):
				attributes_idxs = get_attributes(label + get_slots_offset())
				chosen_attributes = torch.tensor(list(set([v.item() for v in chosen_labels]) & set(attributes_idxs)))
				new_slots = dict()
				slot2attrs = defaultdict(list)
				for i, token in enumerate(tokens):
					if predictions[i][label]:
						#if adjacent tokens have this label, merge
						if i-1 in new_slots:
							# new_slots[i] = torch.stack([new_slots[i-1], token], dim=0)
							new_slots[i] = new_slots[i-1] + [token]
							del new_slots[i-1]
							if chosen_attributes.size(0) > 0 and predictions[i][chosen_attributes].nonzero().size(0) > 0:
								slot2attrs[i] = slot2attrs[i-1] + chosen_attributes[predictions[i][chosen_attributes].nonzero()].squeeze(0).tolist()
								# if not slot2attrs[i]:  # If i-1 was empty, and there are no attributes for this token
								# 	del slot2attrs[i]
							elif slot2attrs[i-1]:
								slot2attrs[i] = slot2attrs[i-1]
								del slot2attrs[i-1]
						else:
							new_slots[i] = [token]
							if chosen_attributes.size(0) > 0 and predictions[i][chosen_attributes].nonzero().size(0) > 0:
								attrs = chosen_attributes[predictions[i][chosen_attributes].nonzero()].squeeze(0).tolist()
								slot2attrs[i] += attrs
				if len(new_slots) > 1 and not is_protected_characteristic(label + get_slots_offset()) and '#' in self.word_to_index:
					# While you can have multiple PCs, for the other slots just merge adding '#'
					keys = list(new_slots.keys())
					sorted(keys)
					for k in keys[1:]:
						new_slots[keys[0]] += [torch.tensor(self.word_to_index['#'], device=self.device)] + new_slots[k]
						del new_slots[k]
						if k in slot2attrs:
							slot2attrs[keys[0]] += slot2attrs[k]
							del slot2attrs[k]
				for k, v in new_slots.items():
					#if any of the merged slots have an attribute, assign the attribute to the whole slot. TODO: What if multiple attributes?
					v = torch.stack(v, dim=0)
					if k in slot2attrs:
						attr_dict = {slot2attrs[k][0]: v[~(v == self.pad_token)].tolist()}
						label2chunks[label.item()] = attr_dict
					else:
						label2chunks[label.item()] = v[~(v == self.pad_token)].tolist()
		# use assigned labels and word ids to generate string
		linearised_tree = ['[', infer_intent_from_slots(list(label2chunks.keys())), ',']
		for k, v in label2chunks.items():
			linearised_tree += ['[', get_ontology_name(k + get_slots_offset()), ',']
			if isinstance(v, dict):
				kk = list(v.keys())[0]
				span = ' '.join([self.index_to_word[idx] for idx in v[kk]])
				linearised_tree += ['[', get_ontology_name(kk + get_slots_offset()), ',', span, ']', ',']
			else:
				span = ' '.join([self.index_to_word[idx] for idx in v])
				linearised_tree += [span, ']', ',']
		linearised_tree = linearised_tree[:-1] + [']']
		return linearised_tree


