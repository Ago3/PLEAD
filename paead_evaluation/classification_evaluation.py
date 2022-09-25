from paead_evaluation import Eval
from paead_info import *
import warnings
from collections import Counter


class ClassificationEval(Eval):
	def __init__(self, corpus, predictions_ids, predictions):
		super().__init__(corpus, predictions_ids, predictions)
		self.available_tests = ['f1_micro', 'f1_macro', 'prf1_per_class', 'ema']
		if 'binary' in corpus.task_name and any([p > 1 for p in self.predictions]):  # if per-rule predictions instead of binary
			self.predictions = [0 if p >= len(HATEFUL_RULES) else 1 for p in self.predictions]
		self.labels = [self.ids_to_instances[p_idx.split('_')[0]][0].label for p_idx in predictions_ids]
		if not isinstance(self.labels[0], int):
			if 'binary' in corpus.task_name:
				self.labels = [0 if self.ids_to_instances[p_idx.split('_')[0]][0].rule == 'nothate' else 1 for p_idx in predictions_ids]
			else:
				self.labels = [RULES.index(self.ids_to_instances[p_idx.split('_')[0]][0].rule) for p_idx in predictions_ids]
		self.task_name = corpus.task_name

	def __run_test__(self, test, is_binary=False, true_class='all', instancefilter=None):
		if is_binary and any([p > 1 for p in self.predictions]):  # if per-rule predictions instead of binary
			predictions = [0 if p >= len(HATEFUL_RULES) else 1 for p in self.predictions]
		else:
			predictions = self.predictions
		if is_binary and 'binary' not in self.task_name:
			labels = [0 if p >= len(HATEFUL_RULES) else 1 for p in self.labels]
		else:
			labels = self.labels
		
		if not true_class == 'all':
			filtered_predictions = []
			filtered_labels = []
			for p, l in zip(predictions, labels):
				hateclass = 'hate' if ((is_binary and l) or (not is_binary and l < len(HATEFUL_RULES))) else 'nothate'
				if hateclass == true_class:
					filtered_predictions.append(p)
					filtered_labels.append(l)
			predictions = filtered_predictions
			labels = filtered_labels

		return self.__aggregate__(test, [labels, predictions])
