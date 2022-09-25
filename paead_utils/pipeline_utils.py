from paead_preprocessing import Corpus
from paead_info import *
from paead_utils import get_slots_offset
from itertools import groupby
import os
import pickle


MEANING_SKETCH_PLACEHOLDER = '<text>'
SPECIAL_TOKENS = ['pad', 'unk']


def get_corpus(task_name, toy, extended_dataset=False, transform=None):
	notfound = False
	if not os.path.exists(LOG_DIR):
		os.makedirs(LOG_DIR)
	if not os.path.exists(CORPUS_DIR):
		os.makedirs(CORPUS_DIR)
		notfound = True
	corpus_file = f'{task_name}_{toy}_corpus.pkl' if not extended_dataset else f'ext_{task_name}_{toy}_corpus.pkl'
	if not USE_ATTRIBUTES: corpus_file = 'noAttributes_' + corpus_file
	if notfound or not os.path.exists(CORPUS_DIR + corpus_file):
		corpus = Corpus(task_name, toy=toy, extended_dataset=extended_dataset)
		with open(CORPUS_DIR + corpus_file, 'wb+') as out:
			pickle.dump(corpus, out)
		if 'classification' not in task_name and not task_name in ['aaa', 'cad']:
			create_vocab_file(corpus, CORPUS_DIR + f'{task_name}_{toy}_vocab.txt')
			create_src_tgt_files(corpus, CORPUS_DIR + f'{task_name}_{toy}')
			corpus.vocab_file = CORPUS_DIR + f'{task_name}_{toy}_vocab.txt'
			corpus.datafiles_prefix = CORPUS_DIR + f'{task_name}_{toy}'
		# return corpus
	else:
		with open(CORPUS_DIR + corpus_file, 'rb+') as infile:
			corpus = pickle.load(infile)
			if 'classification' not in task_name:
				corpus.vocab_file = CORPUS_DIR + f'{task_name}_{toy}_vocab.txt'
				corpus.datafiles_prefix = CORPUS_DIR + f'{task_name}_{toy}'
	return corpus


def create_vocab_file(corpus, filename):
	words = [w for instance in corpus.instances for w in instance.tokenized_text]
	words += TOKENS[get_slots_offset():] + [MEANING_SKETCH_PLACEHOLDER] + SPECIAL_TOKENS # We don't want to generate intent tokens
	words = set(words)
	with open(filename, 'w+') as out:
		for i, w in enumerate(words):
			out.write(f'{w} {i}\n')


def create_src_tgt_files(corpus, filename_prefix):
	for split, split_name in zip(corpus.split_idxs, ['.train', '.val', '.test']):
		with open(filename_prefix + split_name + '.text', 'w+') as src, open(filename_prefix + split_name + '.template', 'w+') as template, open(filename_prefix + split_name + '.text_and_template', 'w+') as text_template, open(filename_prefix + split_name + '.tree', 'w+') as tgt, open(filename_prefix + split_name + '.text_or_template', 'w+') as text_or_template, open(filename_prefix + split_name + '.template_or_tree', 'w+') as template_or_tree, open(filename_prefix + split_name + '.intents', 'w+') as tgt_intents:
			for idx in split:
				instance = corpus.fullids_to_instances[idx]
				src.write(' '.join(instance.tokenized_text) + '\n')
				tgt.write(' '.join(instance.tokenized_label[3:-1]) + '\n')
				meaning_sketch = [w if w in TOKENS else MEANING_SKETCH_PLACEHOLDER for w in instance.tokenized_label[3:-1]]
				meaning_sketch = [x[0] for x in groupby(meaning_sketch)]
				template.write(' '.join(meaning_sketch) + '\n')
				text_template.write(' '.join(instance.tokenized_text) + ' </s> ' + ' '.join(meaning_sketch) + '\n')
				text_or_template.write(' '.join(instance.tokenized_text) + '\n')
				text_or_template.write(' '.join(instance.tokenized_text) + ' </s> ' + ' '.join(meaning_sketch) + '\n')
				template_or_tree.write(' '.join(meaning_sketch) + '\n')
				template_or_tree.write(' '.join(instance.tokenized_label[3:-1]) + '\n')
				tgt_intents.write(instance.tokenized_label[1] + '\n')
