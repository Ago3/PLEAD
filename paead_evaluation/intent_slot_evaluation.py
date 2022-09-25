from paead_evaluation import Eval
from paead_utils import level_to_slot, jaccard_similarity, dict_symmetric_difference, get_productions
from paead_info import USE_ATTRIBUTES
from collections import Counter, defaultdict
from copy import deepcopy
import warnings
import numpy as np


class IntentSlotEval(Eval):
	def __init__(self, corpus, predictions_ids, predictions):
		super().__init__(corpus, predictions_ids, predictions)
		self.available_tests = ['ema_tree', 'ema_intent', 'ema_slot', 'ema_attribute', 'ema_slot_tree', 'ema_jaccard_similarity', 'f1_cfm_per_token_productions', 'f1_cfm_per_chunk_productions', 'f1_cfm_per_token_only_slots_productions', 'f1_cfm_per_chunk_only_slots_productions', 'f1_cfm_top_level_productions']
		if not USE_ATTRIBUTES:
			self.available_tests = [t for t in self.available_tests if 'attribute' not in t]

	def __run_test__(self, test, same_order, true_class='all', instancefilter=None):
		outcomes = []
		for p_idx, prediction in zip(self.predictions_ids, self.predictions):
			instanceList = self.ids_to_instances[p_idx.split('_')[0]]
			if not instanceList:
				continue
			hateclass = 'nothate' if instanceList[0].rule == 'nothate' else 'hate'
			if (not true_class == 'all') and (not hateclass == true_class) and (instancefilter == None):
				continue
			elif instancefilter is not None and (not instancefilter(instanceList[0])):
				continue
			outcome = getattr(IntentSlotEval, f'__{test}_test__')(self, instanceList, prediction, same_order)
			outcomes.append(outcome)
		return self.__aggregate__(test, outcomes)

	# Exact Match Accuracy for full tree, in order or not
	def __ema_tree_test__(self, instanceList, prediction, same_order):
		if same_order:
			return any([instance.tokenized_label == prediction for instance in instanceList])
		prediction_levels = level_to_slot(prediction)
		for instance in instanceList:
			same = True
			instance_levels = level_to_slot(instance.tokenized_label, label=True)
			#If Prediction and Instance don't have the same number of levels, no need to check further
			if not len(prediction_levels.keys()) == len(instance_levels.keys()):
				continue
			for level_idx, level in prediction_levels.items():
				# Prediction and Instance must have the same number of fields per level
				if not len(level) == len(instance_levels[level_idx]):
					same = False
					break
				for node in level:
					# Each node in Prediction.level must be also in Instance.level (don't need to check the other way around)
					if node not in instance_levels[level_idx]:
						same = False
						break
				if not same:
					break
			if same:
				return True
		return False

	# Exact Match Accuracy for intents, in order or not
	def __ema_intent_test__(self, instanceList, prediction, same_order):
		possibleIntents = [[w for w in instance.tokenized_label if 'IN:' in w] for instance in instanceList]
		predictionIntents = [w for w in prediction if 'IN:' in w]
		for intentList in possibleIntents:
			if not len(intentList) == len(predictionIntents):
				continue
			if intentList == predictionIntents:
				return True
			elif not same_order and Counter(intentList) == Counter(predictionIntents):
				return True
		return False

	# Exact Match Accuracy for tree (ignoring top-level intent), in order or not
	def __ema_slot_tree_test__(self, instanceList, prediction, same_order):
		# instances start with the intent by construction. Need to check the prediction though
		prediction_top_level = level_to_slot(prediction)[0]
		if len(prediction_top_level) == 0 or len(prediction_top_level) > 1:
			return False
		if not isinstance(prediction_top_level[0], str) or not prediction_top_level[0].startswith('IN:'):
			return False
		prediction = prediction[3:-1]  #  Remove: '[IN:SOMETHING,' and final ']'
		newInstanceList = []
		for instance in instanceList:
			newInstance = deepcopy(instance)
			newInstance.tokenized_label = instance.tokenized_label[3:-1]
			newInstanceList.append(newInstance)
		return self.__ema_tree_test__(newInstanceList, prediction, same_order)


	def __ema_field_subroutine__(self, instanceList, prediction, same_order, prefix):
		predicted_slots = [w for w in prediction if w.startswith(prefix)]
		instances_slots = [[w for w in instance.tokenized_label if w.startswith(prefix)] for instance in instanceList]
		if same_order:
			return any([predicted_slots == slots for slots in instances_slots])
		predicted_slots_counts = Counter(predicted_slots)
		for slots in instances_slots:
			slots_counts = Counter(slots)
			diff = dict_symmetric_difference(predicted_slots_counts, slots_counts)
			if len(diff) == 0:
				return True
		return False

	# Exact Match Accuarcy for slots, in order or not
	def __ema_slot_test__(self, instanceList, prediction, same_order):
		return self.__ema_field_subroutine__(instanceList, prediction, same_order, 'SL:')

	# Exact Match Accuarcy for attributes, in order or not
	def __ema_attribute_test__(self, instanceList, prediction, same_order):
		return self.__ema_field_subroutine__(instanceList, prediction, same_order, 'ATTR:')

	# Mean Jaccard similarity for full tree (first over tokens, then over slots, finally with the intent), NOT in order
	def __ema_jaccard_similarity_test__(self, instanceList, prediction, same_order=False):
		if same_order:
			warnings.warn('Argument same_order was set to True, but Mean Jaccard Similarity Test doesn\'t take order into account.')
		prediction_levels = level_to_slot(prediction)
		score_per_reference = []
		#Pick a reference
		for instance in instanceList:
			score_per_slot = []
			instance_levels = level_to_slot(instance.tokenized_label, label=True)
			#Match slots (and therefore attributes) between prediction and reference
			n_levels = max(len(prediction_levels), len(instance_levels))
			for level in range(n_levels-1, 0, -1):  # We can skip the Intent for now
				# If this level is missing from either the prediction or the reference, add 0s.
				if (level not in prediction_levels) ^ (level not in instance_levels):
					n_unmachted_slots = len(prediction_levels[level]) if level in prediction_levels else len(instance_levels[level])
					score_per_slot += [0] * n_unmachted_slots
				else:
					slot_occs_pred = Counter([k if isinstance(k, str) else k[0] for k in prediction_levels[level]])
					slot_occs_ref = Counter([k if isinstance(k, str) else k[0] for k in instance_levels[level]])
					for slot in instance_levels[level]:
						if isinstance(slot, str): # It's not a leaf node, no tokens to match
							continue
						# slot_name = slot if isinstance(slot, str) else slot[0]
						slot_name = slot[0]
						if slot_name in slot_occs_pred:  # One or multiple possible matches found: pick the best one
							candidate_scores = []
							for pred_slot in prediction_levels[level]:
								pred_slot_name = pred_slot if isinstance(pred_slot, str) else pred_slot[0]
								if slot_name == pred_slot_name and type(slot) == type(pred_slot):
									#Compute token overlap
									# if isinstance(slot, tuple):
									score = jaccard_similarity([slot[1], pred_slot[1]])  # It's a leaf node
									# else:
									# 	score = 1  # It's not a leaf node, so just match the name
									candidate_scores.append(score)
							if candidate_scores == []:
								score_per_slot.append(0)
							else:
								score_per_slot.append(max(candidate_scores))
					#Penalise for extra slots in predictions, or for slots not predicted
					for k, v in slot_occs_ref.items():
						if k not in slot_occs_pred:
							found = 0
						else:
							found = slot_occs_pred[k]
						if abs(found - v) > 0:
							score_per_slot += [0] * abs(found - v)
					for k, v in slot_occs_pred.items():
						if k in slot_occs_ref:
							continue  # Already penalised for this
						score_per_slot += [0] * abs(v)
			#Average over all slots pairs (non-matched slots count as 0s)
			mean_score_slots = np.mean(score_per_slot) if score_per_slot else 0
			if len(prediction_levels[0]) > 1 or len(prediction_levels[0]) == 0:
				intent_score = 0
			else:
				intent_score = 1 if prediction_levels[0] == instance_levels[0] else 0
			score = np.mean([mean_score_slots, intent_score])
			score_per_reference.append(score)
		return max(score_per_reference)


	def __f1_cfm_productions_test__(self, instanceList, prediction, same_order=False, per_token=True):
		if same_order:
			warnings.warn('Argument same_order was set to True, but Production F1 Test doesn\'t take order into account.')
		prediction_productions = get_productions(prediction, per_token=per_token)
		prediction_counter = Counter(prediction_productions)
		score_per_reference = []
		cfm_per_reference = []
		#Pick a reference
		for instance in instanceList:
			instance_productions = get_productions(instance.tokenized_label, label=True, per_token=per_token)
			instance_counter = Counter(instance_productions)
			key_set = set(list(prediction_counter.keys()) + list(instance_counter.keys()))
			tp, fp, fn = 0, 0, 0
			for k in key_set:
				if k not in prediction_counter:
					fn += instance_counter[k]
				elif k not in instance_counter:
					fp += prediction_counter[k]
				else:
					tp += min(prediction_counter[k], instance_counter[k])
					diff = abs(prediction_counter[k] - instance_counter[k])
					if prediction_counter[k] > instance_counter[k]:
						fp += diff
					elif instance_counter[k] > prediction_counter[k]:
						fn += diff
			f1 = tp / (tp + 0.5 * (fp + fn))
			score_per_reference.append(f1)
			cfm_per_reference.append([tp, fp, fn])
		try: # For human score
			best_ref = np.argmax(score_per_reference)
		except:
			return 0, 0, 0
		return cfm_per_reference[best_ref]


	def __f1_cfm_per_token_productions_test__(self, instanceList, prediction, same_order=False):
		return self.__f1_cfm_productions_test__(instanceList, prediction, same_order, per_token=True)


	def __f1_cfm_per_chunk_productions_test__(self, instanceList, prediction, same_order=False):
		return self.__f1_cfm_productions_test__(instanceList, prediction, same_order, per_token=False)


	def __f1_cfm_only_slots_productions_test__(self, instanceList, prediction, same_order=False, per_token=False):
		if same_order:
			warnings.warn('Argument same_order was set to True, but Production F1 Test doesn\'t take order into account.')
		prediction_productions = get_productions(prediction, per_token=per_token)
		prediction_counter = Counter(prediction_productions)
		slots_to_productions = defaultdict(list)
		for p in prediction_productions:
			if 'SL:' in p[0]:
				slots_to_productions[p[0]].append(p)
		score_per_reference = []
		cfm_per_reference = []
		#Pick a reference
		for instance in instanceList:
			instance_productions = get_productions(instance.tokenized_label, label=True, per_token=per_token)
			instance_counter = Counter(instance_productions)
			key_set = set(list(prediction_counter.keys()) + list(instance_counter.keys()))
			tp, fp, fn = 0, 0, 0
			instance_slots_to_productions = defaultdict(list)
			for p in instance_productions:
				if 'SL:' in p[0]:
					instance_slots_to_productions[p[0]].append(p)
			for k in slots_to_productions.keys():
				if k not in instance_slots_to_productions:
					continue  # We are only evaluating the common slots
				equal = len(set(slots_to_productions[k]).intersection(instance_slots_to_productions[k]))
				if not per_token:
					common = min(len(slots_to_productions[k]), len(instance_slots_to_productions[k]))
				else:
					common = max(len(slots_to_productions[k]), len(instance_slots_to_productions[k]))
				if common - equal > 0:
					if len(slots_to_productions[k]) > equal:
						fp += min(common - equal, len(slots_to_productions[k]) - equal)
					if len(instance_slots_to_productions[k]) > equal:
						fn += min(common - equal, len(instance_slots_to_productions[k]) - equal)
				tp += equal
			f1 = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
			score_per_reference.append(f1)
			cfm_per_reference.append([tp, fp, fn])
		try: # For human score
			best_ref = np.argmax(score_per_reference)
		except:
			return 0, 0, 0
		return cfm_per_reference[best_ref]


	def __f1_cfm_per_chunk_only_slots_productions_test__(self, instanceList, prediction, same_order=False):
		return self.__f1_cfm_only_slots_productions_test__(instanceList, prediction, same_order=False, per_token=False)


	def __f1_cfm_per_token_only_slots_productions_test__(self, instanceList, prediction, same_order=False):
		return self.__f1_cfm_only_slots_productions_test__(instanceList, prediction, same_order=False, per_token=True)


	def __f1_cfm_top_level_productions_test__(self, instanceList, prediction, same_order=False, per_token=False):
		if same_order:
			warnings.warn('Argument same_order was set to True, but Production F1 Test doesn\'t take order into account.')
		prediction_productions = [p for p in get_productions(prediction, per_token=per_token) if 'IN:' in p[0]]
		prediction_counter = Counter(prediction_productions)
		score_per_reference = []
		cfm_per_reference = []
		#Pick a reference
		for instance in instanceList:
			instance_productions = [p for p in get_productions(instance.tokenized_label, label=True, per_token=per_token) if 'IN:' in p[0]]
			instance_counter = Counter(instance_productions)
			key_set = set(list(prediction_counter.keys()) + list(instance_counter.keys()))
			tp, fp, fn = 0, 0, 0
			for k in key_set:
				if k not in prediction_counter:
					fn += instance_counter[k]
				elif k not in instance_counter:
					fp += prediction_counter[k]
				else:
					tp += min(prediction_counter[k], instance_counter[k])
					diff = abs(prediction_counter[k] - instance_counter[k])
					if prediction_counter[k] > instance_counter[k]:
						fp += diff
					elif instance_counter[k] > prediction_counter[k]:
						fn += diff
			f1 = tp / (tp + 0.5 * (fp + fn))
			score_per_reference.append(f1)
			cfm_per_reference.append([tp, fp, fn])
		try: # For human score
			best_ref = np.argmax(score_per_reference)
		except:
			return 0, 0, 0
		return cfm_per_reference[best_ref]


	def __f1_intent_classification_test__(self, instanceList, prediction, same_order):
		pass
