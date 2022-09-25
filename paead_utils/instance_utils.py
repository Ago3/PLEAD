from collections import defaultdict
import string
import re
from paead_info import ONTOLOGY, SLOT_TO_ATTRS, USE_ATTRIBUTES, CAN_COOCCUR, SLOTS
import numpy as np
import itertools
from copy import deepcopy
import re


def title(name):
	return ''.join([word.title() for word in name.split('_')])


def isSpecialToken(token):
	if any(token.startswith(prefix) for prefix in ['IN:', 'SL:', 'ATTR:']): return True
	if token in ['[', ']', ',']: return True
	return False


def level_to_slot(tokenized_tree_string, label=False):
	def rec_step(tree, level_to_slot, current_level=-1):
		if not tree:
			return level_to_slot
		elif tree[0] == '[':
			return rec_step(tree[1:], level_to_slot, current_level+1)
		elif 'IN:' in tree[0]:
			level_to_slot[current_level].append(tree[0])
			return rec_step(tree[1:], level_to_slot, current_level)
		elif 'SL:' in tree[0]:
			if len(tree) < 4 or 'ATTR:' not in tree[3]:
				next_idx = 2   # tree[1] is ','
				while next_idx < len(tree) and not isSpecialToken(tree[next_idx]):
					next_idx += 1
				level_to_slot[current_level].append((tree[0], tree[2:max(3, next_idx)]))
			else:
				level_to_slot[current_level].append(tree[0])
				next_idx = 1
			return rec_step(tree[next_idx:], level_to_slot, current_level)
		elif 'ATTR:' in tree[0]:
			next_idx = 2   # tree[1] is ','
			while next_idx < len(tree) and not isSpecialToken(tree[next_idx]):
				next_idx += 1
			level_to_slot[current_level].append((tree[0], tree[2:max(3, next_idx)]))
			return rec_step(tree[next_idx:], level_to_slot, current_level)
		elif tree[0] == ']':
			return rec_step(tree[1:], level_to_slot, current_level-1)
		elif tree[0] == ',':
			return rec_step(tree[1:], level_to_slot, current_level)
		elif label:
			assert False, f'Unexpected tree node: {tree[0]}'
		else:  # Predictions could just be not well-formed trees
			next_idx = 1
			while next_idx < len(tree) and not isSpecialToken(tree[next_idx]):
				next_idx += 1
			level_to_slot[current_level].append(tuple(tree[0:max(1, next_idx)]))
			return rec_step(tree[next_idx:], level_to_slot, current_level)

	return rec_step(tokenized_tree_string, defaultdict(list))


def get_productions(tokenized_tree_string, per_token=True, label=False):
	# Generators is used as a stack
	def rec_step(tree, production_rules, generators, current_level=-1):
		if not tree:
			return production_rules
		elif tree[0] == '[':
			return rec_step(tree[1:], production_rules, generators, current_level+1)
		elif 'IN:' in tree[0]:
			# No production rule to add
			generators.append(tree[0])
			return rec_step(tree[1:], production_rules, generators, current_level)
		elif 'SL:' in tree[0]:
			production_rules.append((generators[-1], tree[0])) # Add rule: IN -> SLOT
			generators.append(tree[0])
			if len(tree) < 4 or 'ATTR:' not in tree[3]:
				next_idx = 2   # tree[1] is ','
				while next_idx < len(tree) and not isSpecialToken(tree[next_idx]):
					if per_token:
						production_rules.append((generators[-1], tree[next_idx])) # Add rule: SLOT -> token
					next_idx += 1
				if not per_token:
					production_rules.append((generators[-1], ' '.join(tree[2:max(3, next_idx)]))) # Add rule: SLOT -> chunk
			else:
				next_idx = 1
			return rec_step(tree[next_idx:], production_rules, generators, current_level)
		elif 'ATTR:' in tree[0]:
			production_rules.append((generators[-1], tree[0])) # Add rule: SLOT -> ATTR
			generators.append(tree[0])
			next_idx = 2   # tree[1] is ','
			while next_idx < len(tree) and not isSpecialToken(tree[next_idx]):
				if per_token:
					production_rules.append((generators[-1], tree[next_idx])) # Add rule: ATTR -> token
				next_idx += 1
			if not per_token:
				production_rules.append((generators[-1], ' '.join(tree[2:max(3, next_idx)]))) # Add rule: SLOT -> chunk
			return rec_step(tree[next_idx:], production_rules, generators, current_level)
		elif tree[0] == ']':
			if len(generators) > 1 or label:  #Tree might not be well-formed, we want to keep 'root'
				generators.pop()
			return rec_step(tree[1:], production_rules, generators, current_level-1)
		elif tree[0] == ',':
			return rec_step(tree[1:], production_rules, generators, current_level)
		elif label:
			assert False, f'Unexpected tree node: {tree[0]}'
		else:  # Predictions could just be not well-formed trees
			next_idx = 1
			while next_idx < len(tree) and not isSpecialToken(tree[next_idx]):
				if per_token:
					production_rules.append((generators[-1], tree[next_idx]))
				next_idx += 1
			if not per_token:
				production_rules.append((generators[-1], ' '.join(tree[:next_idx]))) # Add rule: SLOT -> chunk
			return rec_step(tree[next_idx:], production_rules, generators, current_level)

	return rec_step(tokenized_tree_string, [], ['root'])


def tokenize(text):
	punctuation = string.punctuation.replace('#', ' ') + '“”'
	return [w for w in re.split(f'[ {punctuation}]', text.replace('#', ' # ')) if w]


def is_protected_characteristic(idx):
	return 'SL:ProtectedCharacteristic' == ONTOLOGY[idx]


def is_attribute(idx):
	return 'ATTR:' in ONTOLOGY[idx]


def has_attributes(idx):
	return ONTOLOGY[idx] in SLOT_TO_ATTRS and USE_ATTRIBUTES


def get_attributes(idx):
	if has_attributes(idx):
		attrs = SLOT_TO_ATTRS[ONTOLOGY[idx]]
		return list(range(ONTOLOGY.index(attrs[0]), ONTOLOGY.index(attrs[-1]) + 1))
	return []


# TODO: What if multiple intents?
def infer_intent_from_slots(slot_ids):
	if not slot_ids:
		return ONTOLOGY[4]
	if isinstance(slot_ids[0], int):
		slot_ids = [s + get_slots_offset() for s in slot_ids]
	else:
		slot_ids = [ONTOLOGY.index(idx) for idx in slot_ids]
	flip_stance = True if 13 in slot_ids else False
	if 9 in slot_ids and 10 in slot_ids:  # Check for hate entities
		return ONTOLOGY[2] if not flip_stance else ONTOLOGY[4]
	if 5 not in slot_ids:  # If no target
		return ONTOLOGY[4]
	if 6 not in slot_ids:  # If no protected characteristic
		return ONTOLOGY[4]
	if 7 in slot_ids:
		return ONTOLOGY[0] if not flip_stance else ONTOLOGY[4]
	if 8 in slot_ids:
		return ONTOLOGY[1] if not flip_stance else ONTOLOGY[4]
	if 11 in slot_ids:
		return ONTOLOGY[3] if not flip_stance else ONTOLOGY[4]
	return ONTOLOGY[4]

def get_ontology_name(idx):
	return ONTOLOGY[idx]

def get_slots_offset():
	for i, v in enumerate(ONTOLOGY):
		if 'IN:' not in v:
			return i

def get_cooccurence_matrix():
	offset = get_slots_offset()
	N = len(ONTOLOGY) - offset
	mtx = np.full((N, N), False, dtype=bool)
	for i in range(N):
		mtx[i, i] = True
	for k, v in CAN_COOCCUR.items():
		row = ONTOLOGY.index(k) - offset
		for vv in v:
			column = ONTOLOGY.index(vv) - offset
			mtx[row, column] = True
	return mtx


def get_slots_onehot(instance):
	slots = [0] * len(SLOTS)
	for k, field in instance.subfields.items():
		if isinstance(field, list):
			if not field:
				continue
			field = field[0]
		slots[SLOTS.index(f'SL:{title(field.field_name)}')] = 1
	return slots


def generate_shuffled_instances(instance, next_available_opinion_id):
	# Need to change label and tokenized_label and ID
	assert instance.tokenized_label, f"Can't generate shuffled instances for this task"
	assert not USE_ATTRIBUTES, "Generation of shuffled instances does not support attributes"
	instance_levels = level_to_slot(instance.tokenized_label, label=True)
	new_instances = []
	for slot_permutation in list(itertools.permutations(instance_levels[1]))[1:]:  # Skip original order
		shuffled_tokenized_label = ['[', instance_levels[0][0], ',']
		shuffled_label = ['[', instance_levels[0][0], ',']
		for slot in slot_permutation:
			shuffled_tokenized_label += ['[', slot[0], ','] + slot[1] + [']', ',']
			shuffled_label += ['[', slot[0], ',', ' '.join(slot[1]), ']', ',']
		shuffled_tokenized_label = shuffled_tokenized_label[:-1] + [']']
		shuffled_label = shuffled_label[:-1] + [']']
		shuffled_label = ''.join(shuffled_label)
		new_instance = deepcopy(instance)
		new_instance.tokenized_label = shuffled_tokenized_label
		new_instance.label = shuffled_label
		new_instance.opinionID = next_available_opinion_id
		next_available_opinion_id += 1
		new_instance.fullID = '_'.join(new_instance.fullID.split('_')[:-1] + [str(new_instance.opinionID)])
		new_instances.append(new_instance)
	return new_instances


# Given a post and a span of text, locate the span in the post.
# Input:
# - p: post
# - s: span
# Output:
# - list of start-end index pairs [(s0, e0), (s1, e1), ...]
def locate_span(p, s, verbose=False):
	if verbose: print('Span: ', s)
	s = annotation_fix(s)
	s = s.replace('#', ' ')
	p = p.replace('#', ' ')
	# pp, ss = p.split(), s.split()
	pp, ss = re.split(r'[^\w]+', p), re.split(r'[^\w]+', s)
	pp = [w for w in pp if w]
	ss = [w for w in ss if w]
	if not all([w in pp for w in ss]) and s in p:
		spans = re.findall(rf'\b\w*?{re.escape(s)}.*?[\b\.,\s]', p + '\n')
		ss = re.split(r'[^\w]+', spans[0])
		ss = [w for w in ss if w]
	assert all([w in pp for w in ss]), f"Word appears in the span '{s}' but not in the post '{p}'"
	# Use dynamic programming to find the minimum number of subspans
	M = [0] * len(pp)
	for i in range(pp.index(ss[0]), len(pp)):
		M[i] = 1
	sol = [M]
	for i, w in enumerate(ss[1:]):
		M2 = [0] * len(pp)
		last = 0
		for j, ref in enumerate(pp):
			if w == ref and j>0 and M[j-1] > 0:
				if pp[j-1] == ss[i]:
					M2[j] = M[j-1]
				else:
					M2[j] = M[j-1] + 1
				last = M2[j]
			else:
				M2[j] = last
		sol.append(M2)
		M = M2
	assignment = []
	col = sol[-1].index(min([v for v in sol[-1] if v]))
	for row in range(len(sol) - 1, -1, -1):
		while col > -1 and sol[row][col] == sol[row][col-1]:
			if pp[col] == ss[row]:
				break
			col -= 1
		assignment.append(col)
		col -= 1
	r = re.compile(r'\w+')
	dic = { i :(m.start(0), m.group(0)) for i, m in enumerate(r.finditer(p))}
	word_idx_output = []
	char_idx_output = []
	s = -1
	e = -1
	for a in assignment[::-1]:
		if s < 0:
			s = a
			e = a
		elif a == e + 1:
			e += 1
		else:
			word_idx_output.append((s, e))
			char_idx_output.append((dic[s][0], dic[e][0] + len(dic[e][1])))
			s = a
			e = a
	word_idx_output.append((s, e+1))
	char_idx_output.append((dic[s][0], dic[e][0] + len(dic[e][1])))
	for se in char_idx_output:
		s, e = se
		if verbose: print('Subspan: ', p[s:e])
	return word_idx_output, char_idx_output


def locate_corpus_spans(corpus):
	for instance in corpus.instances:
		for fieldname, field in instance.subfields.items():
			if not isinstance(field, list):
				field.word_indexes, field.indexes = locate_span(instance.text, field.field_text, False)
				# print(instance.text.split()[field.word_indexes[0][0]: field.word_indexes[0][1]])
			else:
				for f in field:
					f.word_indexes, f.indexes = locate_span(instance.text, f.field_text, False)
	return corpus


def annotation_fix(s):
	if s == "we would love to see more#Murdering":  # Annotator changed the order of the spans
		return "Murdering#we would love to see more"
	elif s == "its simian Malagasy people lo":
		return "its simian Malagasy people lol"
	if s == "i do not think#gays have every right to live and breathe":
		s = "gays have every right to live and breathe#i do not think"
	if s == "don't#love":
		s = "love#don't"
	return s

