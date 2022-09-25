from paead_info import *
from paead_utils import generate_shuffled_instances
from .instance import Instance, InstanceByTask
import json
import csv
from collections import defaultdict
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
import os
import pickle
from copy import deepcopy


class Corpus():
	def __init__(self, task_name=None, toy=False, extended_dataset=False):
		assert task_name and ('classification' in task_name or not extended_dataset), f'Can\'t use extended dataset for a non-classification task ({task_name})'
		self.instances = list()
		self.task_name = task_name
		self.ids_to_instances = defaultdict(list)
		self.fullids_to_instances = {}
		if task_name in ['aaa', 'cad']:
			self.__create_aaa_corpus__()
		elif not extended_dataset:
			self.__create_corpus_with_annotations__(task_name, toy)
		else:
			self.__create_extended_corpus__(task_name, toy)

	def __create_corpus_with_annotations__(self, task_name=None, toy=False):
		annotations_file = ANNOTATED_DATA if not toy else TOY_ANNOTATED_DATA
		print('Creating corpus...')
		with open(annotations_file) as annfile:
			annotations = json.load(annfile)['annotations']
			chunked_instances = self.__get_chunks__(annotations, f'{task_name}_{toy}_chunks.pkl')
			for ann in annotations:
				ann['text'] = ann['text'].replace('\n', ' ').replace('\r', ' ')
				ann['text'] = ' '.join(ann['text'].split())
				if self.task_name is None:
					instance = Instance(ann)
				else:
					instance = InstanceByTask(ann, self.task_name, predicted_chunks=chunked_instances[ann['qid']])
				self.instances.append(instance)
				self.ids_to_instances[instance.qID].append(instance)
				self.fullids_to_instances[instance.fullID] = instance
		self.split_idxs = self.__get_ids__(toy, extended_dataset=False)

	def __create_extended_corpus__(self, task_name=None, toy=False):
		dataset_file = EXTENDED_DATASET_FILE if not toy else EXTENDED_TOY_DATASET_FILE
		print('Creating corpus...')
		with open(dataset_file, 'r') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			next(csvreader)  # Skip header
			for row in csvreader:
				qID, text, rule = row
				text = text.replace('\n', ' ').replace('\r', ' ')
				text = ' '.join(text.split())
				instance_json = {
					'text': text
				}
				annotation_json = {
					'qid': qID,
					'copyid': 0,
					'rule': rule
				}
				instance = InstanceByTask([instance_json, annotation_json], task_name=self.task_name)
				self.instances.append(instance)
				self.ids_to_instances[instance.qID].append(instance)
				self.fullids_to_instances[instance.fullID] = instance
		self.split_idxs = self.__get_ids__(toy, extended_dataset=True)

	def __create_aaa_corpus__(self):
		print('Creating corpus...')
		qids = 0
		self.split_idxs = []
		files = AAA_FILES if self.task_name == 'aaa' else CAD_FILES
		main_dir = AAA_DIR if self.task_name == 'aaa' else CAD_DIR
		for dataset_file in files:
			print('Current file: ', dataset_file)
			current_split = []
			with open(main_dir + dataset_file, 'r') as csvfile:
				csvreader = csv.reader(csvfile, delimiter='\t')
				for row in csvreader:
					text, label = row
					text = ' '.join(text.split())
					instance_json = {
						'text': text
					}
					rule = 'nothate' if not int(label) else 'hate'
					annotation_json = {
						'qid': qids,
						'copyid': 0,
						'rule': rule
					}
					qids +=1
					instance = InstanceByTask([instance_json, annotation_json], task_name='binary_classification')
					self.instances.append(instance)
					self.ids_to_instances[instance.qID].append(instance)
					self.fullids_to_instances[instance.fullID] = instance
					current_split.append(instance.fullID)
			self.split_idxs.append(current_split)

	def __get_ids__(self, toy, extended_dataset=False):
		if toy:
			self.only_hate_train_ids = [k for k,v in self.fullids_to_instances.items() if v.rule in HATEFUL_RULES]
			return [list(self.fullids_to_instances.keys())] * 3
		all_split_ids = []
		for split in ['train', 'val', 'test']:
			with open(SPLIT_FILES[split] if not extended_dataset else EXTENDED_SPLIT_FILES[split], 'r') as f:
				split_ids = ['_'.join(line.strip().split(',')) for line in f.readlines()]
				all_split_ids.append(split_ids)
				if split == 'train':
					self.only_hate_train_ids = [k for k in split_ids if self.fullids_to_instances[k].rule in HATEFUL_RULES]
		return all_split_ids

	def get_split_idxs(self):
		return self.split_idxs

	def __check_cached_chunks__(self, chunks_file):
		chunks = None
		if os.path.exists(CORPUS_DIR + chunks_file):
			with open(CORPUS_DIR + chunks_file, 'rb') as cache:
				chunks = pickle.load(cache)
		return chunks

	def __get_chunks__(self, dataset, chunks_file):
		chunks = self.__check_cached_chunks__(chunks_file)
		instances_to_chunk = {}
		tagger = SequenceTagger.load("flair/chunk-english-fast")
		for annotation in dataset:
			qID = annotation['qid']
			text = annotation['text']
				# qID, _, text, _, _, _ = row
			text = text.replace('\n', ' ').replace('\r', ' ')
			text = ' '.join(text.split())
			instances_to_chunk[qID] = text
		if chunks:
			for k in chunks.keys():
				del instances_to_chunk[k]
		chunked_instances = {}
		keys = list(instances_to_chunk.keys())
		if len(keys) > 0:
			sentences = [Sentence(instances_to_chunk[k]) for k in keys]
			tagger.predict(sentences)
			predicted_chunks = [[span['text'] for span in sentence.to_dict(tag_type='np')['entities']] for sentence in sentences]
			for k, v in zip(keys, predicted_chunks):
				chunked_instances[k] = v
		if chunks:
			chunked_instances.update(chunks)
		with open(CORPUS_DIR + chunks_file, 'wb+') as cache:
			pickle.dump(chunked_instances, cache)
		return chunked_instances

	def extend_with_shuffled_instances(self, criterium):
		new_instances = []
		for instance in self.instances:
			if instance.fullID in self.split_idxs[0] and criterium(instance):
				# find next free opinionID
				current_id = instance.fullID
				while current_id in self.fullids_to_instances:
					ids = current_id.split('_')
					current_id = '_'.join(ids[:-1] + [str(int(ids[-1]) + 1)])
				# generate shuffled instances
				additional_instances = generate_shuffled_instances(instance, int(current_id.split('_')[-1]))
				new_instances += additional_instances
		for instance in new_instances:
			self.ids_to_instances[instance.qID].append(instance)
			self.fullids_to_instances[instance.fullID] = instance
			# Add new instance to training set
			self.split_idxs[0].append(instance.fullID)
		self.instances += new_instances
		print('Training set extended: ', len(self.split_idxs[0]))

	def check_and_extend_with_shuffled_instances(self):
		assert USE_SHUFFLING_AUGMENTATION in [None, 'negstance', 'nothate', 'all'], f"Value '{USE_SHUFFLING_AUGMENTATION}'' is not supported for shuffling augmentation"
		if not USE_SHUFFLING_AUGMENTATION: return
		if USE_SHUFFLING_AUGMENTATION == 'negstance':
			criterium = (lambda x: 'SL:NegativeStance' in x.tokenized_label)
		elif USE_SHUFFLING_AUGMENTATION == 'nothate':
			criterium = (lambda x: x.rule == 'nothate')
		else:
			criterium = (lambda x: True)
		self.extend_with_shuffled_instances(criterium)

	def check_and_replace_with_augmented(self):
		if not USE_AUGMENTED:
			return self
		with open(CORPUS_DIR + AUGMENTED_CORPUS_FILE, 'rb') as f:
			return pickle.load(f)

	def get_human_score(self, tests):
		assert 'intent_and_slot_filling' == self.task_name, f'Error: function get_human_score does not support task {self.task_name}'
		from paead_evaluation import IntentSlotEval
		import random
		test_idxs = self.split_idxs[2]
		predictions = []
		id2pred = {}
		new_corpus = deepcopy(self)
		for idx in test_idxs:
			idx = idx.split('_')[0]
			if idx in id2pred:
				predictions.append(id2pred[idx])  # When you encounter different copies of same id
			else:
				annotator_idx = random.randint(0, len(self.ids_to_instances[idx]) - 1)
				instance = self.ids_to_instances[idx][annotator_idx]
				predictions.append(instance.tokenized_label)
				id2pred[idx] = instance.tokenized_label
				new_corpus.ids_to_instances[idx].pop(annotator_idx)
		taskEval = IntentSlotEval(new_corpus, test_idxs, predictions)
		taskEval.eval(tests)
		taskEval.eval(tests, true_class='hate')
		taskEval.eval(tests, true_class='nothate')
