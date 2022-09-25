import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn

import random
import itertools
from collections import defaultdict

from paead_info import *
from paead_evaluation import ClassificationEval


class SlotsToIntentModel(pl.LightningModule):
	# Instantiate the model
	def __init__(self, corpus=None):
		super().__init__()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.slots_to_slots = nn.Linear(len(SLOTS), len(SLOTS)).to(device)
		self.slots_to_rule = nn.Linear(len(SLOTS), len(RULES)).to(device)
		self.learning_rate = S2I_LEARNING_RATE
		self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 3.0, 1.0]))
		self.criterion_binary = torch.nn.BCELoss()
		self.corpus = corpus  # Needed for validation
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)

	def from_pretrained(checkpoint_path):
		slots2intent = SlotsToIntentModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
		return slots2intent

	def visualise(self):
		from matplotlib import pyplot as plt
		tensor = self.slots_to_slots.weight.detach()
		tensor = 2 * ((tensor - tensor.min()) / (tensor.max() - tensor.min())) - 1
		tensor2 = self.slots_to_rule.weight.detach()
		tensor2 = 2 * ((tensor2 - tensor2.min()) / (tensor2.max() - tensor2.min())) - 1
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(tensor, interpolation='nearest')
		fig.colorbar(cax)
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		cax2 = ax2.matshow(tensor2, interpolation='nearest')
		fig2.colorbar(cax2)
		plt.show()

		def visualise_test(test_input, test_input2, title):
			with torch.no_grad():
				test_output = self(test_input)
				fig, (ax, ax2) = plt.subplots(2)
				ax.set_title(title)
				test_output = test_output.unsqueeze(0)
				test_output = (test_output - test_output.min()) / (test_output.max() - test_output.min())
				cax = ax.matshow(test_output, interpolation='nearest')
				fig.colorbar(cax, location='top')
				test_output2 = self(test_input2)
				ax2.set_title(f'{title}, Neg Stance')
				test_output2 = test_output2.unsqueeze(0)
				test_output2 = (test_output2 - test_output2.min()) / (test_output2.max() - test_output2.min())
				cax2 = ax2.matshow(test_output2, interpolation='nearest')
				plt.tick_params(left=False)
				plt.show()

		# Create input with target, PC and comparison
		test_input = torch.tensor([1, 1, 0, 1, 0, 0, 0, 0, 0]).float()
		# Create input with target, PC, comparison and negative stance
		test_input2 = torch.tensor([1, 1, 0, 1, 0, 0, 0, 0, 1]).float()
		visualise_test(test_input, test_input2, 'T, PC, Comparison')
		# Create input with target, PC and derogation
		test_input = torch.tensor([1, 1, 0, 0, 0, 0, 1, 0, 0]).float()
		# Create input with target, PC, derogation and negative stance
		test_input2 = torch.tensor([1, 1, 0, 0, 0, 0, 1, 0, 1]).float()
		visualise_test(test_input, test_input2, 'T, PC, Derogation')
		# Create input with target, PC and threatening
		test_input = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0]).float()
		# Create input with target, PC, threatening and negative stance
		test_input2 = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 1]).float()
		visualise_test(test_input, test_input2, 'T, PC, Threatening')
		# Create input with hate entity and support for hate crimes
		test_input = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0, 0]).float()
		# Create input with hate entity,support for hate crimes negative stance
		test_input2 = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0, 1]).float()
		visualise_test(test_input, test_input2, 'Hate Entity, Support for Hate Crimes')


	# Do a forward pass through the model
	def forward(self, slots_onehot):
		# Find logical ANDs of slots
		slot_constraints = self.relu(self.slots_to_slots(slots_onehot))  # [batch, n_slot_tokens]
		# Find logical ORs on combinations of slots
		intent_scores = self.slots_to_rule(slot_constraints)  # [batch, n_intents]
		return intent_scores
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
		return optimizer

	def training_step(self, batch, batch_idx):
		intent_scores = self(batch['slots_one_hot'])  # [batch, n_intents]		
		# Calculate the loss on the intent
		intent_logits = self.softmax(intent_scores)  # [batch, n_intents]
		intent_loss = self.criterion(intent_logits, batch['intent'])
		# Calculate the loss on the binary intent
		binary_intent_logits = intent_logits[:, :-1].sum(dim=1)  # [batch]
		binary_intent_loss = self.criterion_binary(binary_intent_logits, batch['binary_intent'].float())

		losses = {'loss': (S2I_INTENT_LOSS_WEIGHT * intent_loss) + (S2I_BINARY_INTENT_LOSS_WEIGHT * binary_intent_loss), 'intent_loss': intent_loss, 'binary_intent_loss': binary_intent_loss}

		return losses

	def training_epoch_end(self, losses):
		aggregated_losses = defaultdict(int)
		for d in losses:
			for k, v in d.items():
				aggregated_losses[k] += v
		aggregated_losses = {k: v / len(losses) for k, v in aggregated_losses.items()}
		print('\n', aggregated_losses, '\n')

	def validation_step(self, batch, batch_idx):
		intent_scores = self(batch['slots_one_hot'])  # [batch, n_intents]		
		intent_logits = self.softmax(intent_scores)  # [batch, n_intents]
		predictions = intent_logits.argmax(dim=1)  # [batch]
		predictions = [p.item() for p in predictions]
		predictions_ids = batch['fullID']

		return predictions_ids, predictions

	def validation_epoch_end(self, outs):
		predictions_ids, predictions = list(zip(*outs))
		predictions_ids = list(itertools.chain(*predictions_ids))
		predictions = list(itertools.chain(*predictions))

		taskEval = ClassificationEval(self.corpus, predictions_ids, predictions)
		prod_f1 = taskEval.eval(tests=['f1_micro'], arg=False)[0]

		metrics = {'val_prod_f1': prod_f1}
		
		self.log_dict(metrics)

		return metrics

	def test_step(self, batch, batch_idx):
		return self.validation_step(batch, batch_idx)

	def test_epoch_end(self, outs):
		predictions_ids, predictions = list(zip(*outs))
		predictions_ids = list(itertools.chain(*predictions_ids))
		predictions = list(itertools.chain(*predictions))

		taskEval = ClassificationEval(self.corpus, predictions_ids, predictions)

		scores = taskEval.eval(tests=None, arg=False, verbose=True)
		metrics = {k: s for k, s in zip(taskEval.available_tests, scores)}

		# self.log_dict(metrics)
		return metrics
