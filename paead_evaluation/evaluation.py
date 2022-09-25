from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
from paead_info import *


class Eval:
	def __init__(self, corpus, predictions_ids, predictions):
		assert len(predictions_ids) == len(predictions), f'Expected {len(predictions_ids)} predictions, got {len(predictions)}'
		self.instances = corpus.instances
		self.ids_to_instances = corpus.ids_to_instances
		self.predictions_ids = predictions_ids
		self.predictions = predictions

	def eval(self, tests=None, arg=False, verbose=True, true_class='all', instancefilter=None):  # if tests == None, just run all the tests
		if not tests: tests = self.available_tests
		scores = []
		for test in tests:
			assert test in self.available_tests, f'Test {test} is not supported'
			score = self.__run_test__(test, arg, true_class, instancefilter)
			scores.append(score)
			if verbose:
				if isinstance(score, tuple):
					score = zip(*score)
					metric = ['precision', 'recall', 'f1']
					for i, ss in enumerate(score):
						c = RULES[i] if not arg else ['nothate', 'hate'][i]
						for m, s in zip(metric, ss):
							print(f'Score for test {m}, class {c}: {round(s,4)}')
						print()
				else:
					print(f'Score for test {test}: {score}')
		return scores

	def __aggregate__(self, test, outcomes):
		if not outcomes: return 0
		if 'ema' in test:
			if isinstance(outcomes[0], list):
				outcomes = [1 if l == p else 0 for p, l in zip(*outcomes)]
			return sum(outcomes) / len(outcomes)
		elif 'f1_micro' in test:
			labels, predictions = outcomes
			print(labels[:10])
			print(predictions[:10])
			return f1_score(labels, predictions, average='micro')
		elif 'f1_macro' in test:
			labels, predictions = outcomes
			return f1_score(labels, predictions, average='macro')
		elif 'prf1_per_class' in test:
			labels, predictions = outcomes
			return precision_recall_fscore_support(labels, predictions, labels=list(range(max(labels) + 1)), average=None)[:-1]
		elif 'f1_cfm' in test:
			scores = list(zip(*outcomes))
			tp, fp, fn = sum(scores[0]), sum(scores[1]), sum(scores[2])
			f1 = tp / max((tp + 0.5 * (fp + fn)), 0.000001)
			return f1
		else:
			assert False, f'No aggregation method found for {test}'
