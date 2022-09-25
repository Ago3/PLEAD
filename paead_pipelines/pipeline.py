import argparse
import torch
import random
import subprocess
import os
from paead_info import *
from paead_utils import *
from paead_preprocessing import Corpus
from paead_evaluation import IntentSlotEval, ClassificationEval


def pipeline_intent_and_slot_filling(model_name, corpus, seed):
	corpus.split_idxs[0] = [idx for idx in corpus.split_idxs[0] if 'acl' not in idx.split('_')[-1]]

	parser = argparse.ArgumentParser()
	if model_name in ['token_tagging_embeddings']:
		parser.add_argument('--max-epochs', type=int, default=TT_MAX_EPOCHS)
		parser.add_argument('--batch-size', type=int, default=TT_BATCH_SIZE)
		parser.add_argument('--seed', type=int, default=1)
		parser.add_argument('--name', type=str, default='exp')
		args = parser.parse_args()
		from paead_dataset import TokenTaggingDataset as Dataset
		from paead_models import TokenTaggingModel as Model
		from paead_pipelines.token_tagging import train, predict
		embedding_size = TT_EMBEDDING_SIZE
		resfile = TT_RES_FILE
		exp_name = TT_EXPERIMENT_NAME
	elif model_name in ['litbart']:
		from paead_dataset import LitBARTDataModule as Dataset
		from paead_models import LitBART as Model
		from paead_pipelines.litbart import train
		model = Model(corpus, seed)
		summary_data = Dataset(corpus, corpus.get_split_idxs())
		trainer = train(model, summary_data)
		trainer.test(ckpt_path="best", dataloaders=[summary_data.test_dataloader()])
		return
	elif model_name in ['litmsbart']:
		from paead_dataset import LitBARTDataModule as Dataset
		from paead_models import LitMSBART as Model
		from paead_pipelines.litbart import train
		model = Model(corpus, LMSB_PREDICTIONS_FILE, LMSB_RES_FILE, LMSB_EXPERIMENT_NAME, seed)
		summary_data = Dataset(corpus, corpus.get_split_idxs(), hate_pretraining=LMSB_HATE_PRETRAIN)
		trainer = train(model, summary_data)
		trainer.test(ckpt_path="best", dataloaders=[summary_data.test_dataloader()])
		return
	elif model_name in ['litmsbart_with_slot2intent']:
		from paead_dataset import LitBARTDataModule as Dataset
		from paead_models import LitMSBARTwithSlot2Intent as Model
		from paead_pipelines.litbart import train
		model = Model(corpus, LMSBwS2I_PREDICTIONS_FILE, LMSBwS2I_RES_FILE, LMSBwS2I_EXPERIMENT_NAME, seed)
		summary_data = Dataset(corpus, corpus.get_split_idxs(), hate_pretraining=LMSBwS2I_HATE_PRETRAIN)
		trainer = train(model, summary_data)
		trainer.test(ckpt_path="best", dataloaders=[summary_data.test_dataloader()])
		return
	elif model_name in ['bert_token_tagging']:
		from paead_dataset import LitBARTTagDataModule as Dataset
		from paead_models import LitBERTTag as Model
		from paead_pipelines.litbart import train
		corpus = locate_corpus_spans(corpus)
		summary_data = Dataset(corpus, corpus.get_split_idxs(), hate_pretraining=BERTTAG_HATE_PRETRAIN)
		model = Model(corpus, seed, summary_data.datasets[0].OvO)
		for d in summary_data.datasets:
			d.tokenizer = model.tokenizer
		trainer = train(model, summary_data)
		trainer.test(ckpt_path="best", dataloaders=[summary_data.test_dataloader()])
		return


	split_ids = corpus.get_split_idxs()
	is_training_set = [True, False, False]
	datasets = [Dataset(args, corpus, split_idxs, is_training_set[idx]) for idx, split_idxs in enumerate(split_ids)]
	if 'embeddings' in model_name:
		embeddings = load_embeddings(embedding_size, word_to_index=datasets[0].word_to_index, pad_token=datasets[0].pad_token)
	else:
		embeddings = None
	model = Model(datasets[0], embeddings)
	val_dataset = datasets[1] if len(datasets) == 3 else datasets[0]
	test_dataset = datasets[2] if len(datasets) == 3 else datasets[0]
	train(datasets[0], val_dataset, model, args, seed)
	predictions, predictions_ids = predict(test_dataset, model, seed)
	taskEval = IntentSlotEval(corpus, predictions_ids, predictions)
	scores = []
	scores.append(taskEval.eval(tests=None, arg=False))
	scores.append(taskEval.eval(tests=None, arg=False, verbose=False, true_class='hate'))
	scores.append(taskEval.eval(tests=None, arg=False, verbose=False, true_class='nothate'))
	for subfix, sub_scores in zip(['', '.hate', '.nothate'], scores):
		with open(resfile + subfix, 'a+') as out:
			sub_scores = [str(s) for s in sub_scores]
			out.write(f'{exp_name}\t{seed}\t' + '\t'.join(sub_scores) + '\n')


def pipeline_classification(model_name, corpus, seed):
	parser = argparse.ArgumentParser()
	if model_name in ['roberta']:
		parser.add_argument('--max-epochs', type=int, default=RM_MAX_EPOCHS)
		parser.add_argument('--batch-size', type=int, default=RM_BATCH_SIZE)
		parser.add_argument('--seed', type=int, default=1)
		parser.add_argument('--name', type=str, default='exp')
		args = parser.parse_args()
		from paead_dataset import RobertaDataset as Dataset
		from paead_models import RobertaModel as Model
		from paead_pipelines.roberta import train, predict
		resfile = RM_RES_FILE
		exp_name = RM_EXPERIMENT_NAME

	split_ids = corpus.get_split_idxs()
	is_training_set = [True, False, False]
	datasets = [Dataset(args, corpus, split_idxs, is_training_set[idx]) for idx, split_idxs in enumerate(split_ids)]
	model = Model(datasets[0])
	val_dataset = datasets[1] if len(datasets) == 3 else datasets[0]
	test_dataset = datasets[2] if len(datasets) == 3 else datasets[0]
	if not 'zero_shot' in model_name:
		train(datasets[0], val_dataset, model, args, seed)
	predictions, predictions_ids = predict(test_dataset, model, zeroshot=('zero_shot' in model_name), seed=seed)
	taskEval = ClassificationEval(corpus, predictions_ids, predictions)
	print('\nMulti-class Evaluation:')
	scores = taskEval.eval(tests=None, arg=False)
	with open(resfile, 'a+') as out:
		scores = [str(s) for s in scores]
		out.write(f'{exp_name}\t{seed}\t' + '\t'.join(scores) + '\n')
	print('\nBinary Evaluation:')
	scores = taskEval.eval(tests=None, arg=True)


def pipeline_aaa(model_name, corpus, aaa_corpus, seed):
	parser = argparse.ArgumentParser()
	if model_name in ['roberta']:
		parser.add_argument('--max-epochs', type=int, default=RM_MAX_EPOCHS)
		parser.add_argument('--batch-size', type=int, default=RM_BATCH_SIZE)
		parser.add_argument('--seed', type=int, default=1)
		parser.add_argument('--name', type=str, default='exp')
		args = parser.parse_args()
		from paead_dataset import RobertaDataset as Dataset
		from paead_models import RobertaModel as Model
		from paead_pipelines.roberta import train, predict
		aaa_files_dir = RM_AAA_FILES

	elif model_name in ['token_tagging_embeddings']:
		parser.add_argument('--max-epochs', type=int, default=TT_MAX_EPOCHS)
		parser.add_argument('--batch-size', type=int, default=TT_BATCH_SIZE)
		parser.add_argument('--seed', type=int, default=1)
		parser.add_argument('--name', type=str, default='exp')
		args = parser.parse_args()
		from paead_dataset import TokenTaggingDataset as Dataset
		from paead_models import TokenTaggingModel as Model
		from paead_pipelines.token_tagging import train, predict
		embedding_size = TT_EMBEDDING_SIZE
		resfile = TT_RES_FILE
		exp_name = TT_EXPERIMENT_NAME
		aaa_files_dir = TT_AAA_FILES

	elif model_name in ['litbart']:
		from paead_dataset import LitBARTDataModule as Dataset
		from paead_models import LitBART as Model
		from paead_pipelines.litbart import train
		from paead_info import LB_AAA_CKPT
		model = Model(aaa_corpus, seed)
		summary_data = Dataset(aaa_corpus, aaa_corpus.get_split_idxs(), hate_pretraining=LB_HATE_PRETRAIN)
		trainer = train(model, summary_data, disable_training=True)
		# Find best model for this seed
		LB_AAA_CKPT = LB_AAA_CKPT.replace('SEED', str(seed))
		ckpts = [f for f in os.listdir(LB_AAA_CKPT)]
		LB_AAA_CKPT += ckpts[0]
		trainer.test(model=model, ckpt_path=LB_AAA_CKPT, dataloaders=summary_data.aaa_test_dataloaders())
		return

	elif model_name in ['litmsbart']:
		from paead_dataset import LitBARTDataModule as Dataset
		from paead_models import LitMSBART as Model
		from paead_pipelines.litbart import train
		from paead_info import LMSB_AAA_CKPT
		model = Model(aaa_corpus, LMSB_AAA_ANSWER_FILES, LMSB_RES_FILE, LMSB_EXPERIMENT_NAME, seed)
		summary_data = Dataset(aaa_corpus, aaa_corpus.get_split_idxs(), hate_pretraining=LMSB_HATE_PRETRAIN)
		trainer = train(model, summary_data, disable_training=True)
		LMSB_AAA_CKPT = LMSB_AAA_CKPT.replace('SEED', str(seed - 1))
		ckpts = [f for f in os.listdir(LMSB_AAA_CKPT)]
		LMSB_AAA_CKPT += ckpts[0]
		trainer.test(model=model, ckpt_path=LMSB_AAA_CKPT, dataloaders=summary_data.aaa_test_dataloaders())
		return

	elif model_name in ['litmsbart_with_slot2intent']:
		from paead_dataset import LitBARTDataModule as Dataset
		from paead_models import LitMSBARTwithSlot2Intent as Model
		from paead_pipelines.litbart import train
		from paead_info import LMSBwS2I_AAA_CKPT
		model = Model(aaa_corpus, LMSBwS2I_AAA_ANSWER_FILES, LMSBwS2I_RES_FILE, LMSBwS2I_EXPERIMENT_NAME, seed)
		summary_data = Dataset(aaa_corpus, aaa_corpus.get_split_idxs(), hate_pretraining=LMSBwS2I_HATE_PRETRAIN)
		trainer = train(model, summary_data, disable_training=True)
		LMSBwS2I_AAA_CKPT = LMSBwS2I_AAA_CKPT.replace('SEED', str(seed - 1))
		ckpts = [f for f in os.listdir(LMSBwS2I_AAA_CKPT)]
		LMSBwS2I_AAA_CKPT += ckpts[0]
		trainer.test(model=model, ckpt_path=LMSBwS2I_AAA_CKPT, dataloaders=summary_data.aaa_test_dataloaders())
		return

	elif model_name in ['bert_token_tagging']:
		from paead_dataset import LitBARTTagDataModule as Dataset
		from paead_models import LitBERTTag as Model
		from paead_pipelines.litbart import train
		from paead_info import BERTTAG_AAA_CKPT
		summary_data = Dataset(aaa_corpus, aaa_corpus.get_split_idxs(), hate_pretraining=BERTTAG_HATE_PRETRAIN)
		model = Model(aaa_corpus, seed, summary_data.datasets[0].OvO)
		for d in summary_data.datasets:
			d.tokenizer = model.tokenizer
		trainer = train(model, summary_data, disable_training=True)
		BERTTAG_AAA_CKPT = BERTTAG_AAA_CKPT.replace('SEED', str(seed - 1))
		ckpts = [f for f in os.listdir(BERTTAG_AAA_CKPT)]
		BERTTAG_AAA_CKPT += ckpts[0]
		trainer.test(model=model, ckpt_path=BERTTAG_AAA_CKPT, dataloaders=summary_data.aaa_test_dataloaders())
		return

	split_ids = corpus.get_split_idxs()
	is_training_set = [True, False, False]
	datasets = [Dataset(args, corpus, split_idxs, is_training_set[idx]) for idx, split_idxs in enumerate(split_ids)]
	if 'embeddings' in model_name:
		embeddings = load_embeddings(embedding_size, word_to_index=datasets[0].word_to_index, pad_token=datasets[0].pad_token)
		model = Model(datasets[0], embeddings)
	else:
		model = Model(datasets[0])
	aaa_datasets = [Dataset(args, aaa_corpus, split_idxs, False) for split_idxs in aaa_corpus.get_split_idxs()]
	for testname, testset in zip(AAA_FILES, aaa_datasets):
		print('Current test: ', testname)
		with open(aaa_files_dir + f'{seed}_' + testname, 'w+') as out:
			if model_name == 'token_tagging_embeddings':
				predictions, predictions_ids = predict(testset, model, seed)
				predictions = [0 if 'IN:NotHateful' in p else 1 for p in predictions]
			else:
				predictions, predictions_ids = predict(testset, model, zeroshot=('zero_shot' in model_name), seed=seed)
			if any([p > 1 for p in predictions]):  # if per-rule predictions instead of binary
				predictions = [0 if p >= len(HATEFUL_RULES) else 1 for p in predictions]
			for p_id, p in zip(predictions_ids, predictions):
				instance = aaa_corpus.fullids_to_instances[p_id]
				label = 0 if instance.rule == 'nothate' else 1
				out.write(f'{instance.text}\t{label}\t{p}\n')


def pipeline_functionality_tests(corpus):
	functionality_tests(corpus)


def run_pipeline(task_name, model_name, toy=False, extended_dataset=False, seed=None):
	assert task_name in AVAILABLE_TASKS, f'{task_name} is not a valid task'
	assert model_name in AVAILABLE_MODELS_BY_TASK[task_name], f'{model_name} is not supported for task {task_name}'
	torch.manual_seed(1 if not seed else seed)
	random.seed(1 if not seed else seed)	
	if task_name == 'intent_and_slot_filling':
		corpus = get_corpus(task_name=task_name, toy=toy, transform=None)
		pipeline_intent_and_slot_filling(model_name, corpus, seed)
	if task_name in ['binary_classification', 'classification']:
		corpus = get_corpus(task_name=task_name, toy=toy, extended_dataset=extended_dataset, transform=None)
		pipeline_classification(model_name, corpus, seed)
	if task_name in ['aaa']:
		corpus = get_corpus(task_name='binary_classification', toy=toy, extended_dataset=extended_dataset)
		aaa_corpus = get_corpus(task_name=task_name, toy=toy, extended_dataset=extended_dataset)
		pipeline_aaa(model_name, corpus, aaa_corpus, seed)
	if task_name == 'slots_to_intent':
		corpus = get_corpus(task_name='intent_and_slot_filling', toy=toy)
		pipeline_slots_to_intent(model_name, corpus)
	if task_name == 'functionality_tests':
		corpus = get_corpus(task_name='intent_and_slot_filling', toy=toy)
		pipeline_functionality_tests(corpus)
