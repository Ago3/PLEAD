from paead_info import *
import os
import torch
import wget
import zipfile
import numpy as np


def save_checkpoint(model, modeldir, modelfile, epoch, valid_score, seed=None):
	if not os.path.exists(LOG_DIR):
		os.makedirs(LOG_DIR)
	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	if seed:
		modeldir += f'{seed}/'
		if not os.path.exists(modeldir):
			os.makedirs(modeldir)
	modelfile = modeldir + modelfile.split('/')[-1]
	model_dict = model.state_dict()
	checkpoint = {
		'model_state_dict': model_dict,
		'epoch': epoch,
		'valid_score': valid_score
	}
	torch.save(checkpoint, modelfile)


def load_checkpoint(model, modeldir, modelfile, seed=None):
	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
		if seed:
			modeldir += f'{seed}/'
			os.makedirs(modeldir)
		return model, 0, -1
	if seed:
		modeldir += f'{seed}/'
	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	print('\n\n\nLoading from: ', modeldir)
	files = [f for f in os.listdir(modeldir) if not f.startswith('.')]
	modelfile_parameters = modelfile.split('/')[-1]
	for filename in files:
		if modelfile_parameters == filename:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			checkpoint = torch.load(modeldir + filename, map_location=device)
			print('Loading model from epoch: ', checkpoint['epoch'])
			model.load_state_dict(checkpoint['model_state_dict'])
			return model, checkpoint['epoch'], checkpoint['valid_score']
	return model, 0, -1


def download_embeddings():
	if not os.path.exists(EMBEDDINGS_PATH):
		print('Downloading Glove embeddings...')
		os.makedirs(EMBEDDINGS_PATH)
		filename = wget.download('http://nlp.stanford.edu/data/glove.6B.zip', out=EMBEDDINGS_PATH)
		with zipfile.ZipFile(EMBEDDINGS_PATH + 'glove.6B.zip', 'r') as zip_ref:
			zip_ref.extractall(EMBEDDINGS_PATH)
		print('Done.')
	else:
		print('Found previously downloaded embeddings.')


def load_embeddings(embedding_dim, word_to_index, pad_token):
	download_embeddings()
	vocab,embeddings = [],[]
	with open(EMBEDDINGS_PATH + f'glove.6B.{embedding_dim}d.txt','rt') as fi:
		full_content = fi.read().strip().split('\n')
	for i in range(len(full_content)):
		i_word = full_content[i].split(' ')[0]
		i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
		if i_word in word_to_index:
			vocab.append(i_word)
			embeddings.append(i_embeddings)
	pretrained_embeddings = np.array(embeddings)
	mean_emb = np.mean(pretrained_embeddings, axis=0, keepdims=True)
	embeddings = np.repeat(mean_emb, len(word_to_index), axis=0)
	copies = 0
	for word, index in word_to_index.items():
		if word in vocab:
			copies += 1
			idx = vocab.index(word)
			embeddings[index, :] = pretrained_embeddings[idx, :]
	embeddings[pad_token, :] = np.zeros((embedding_dim))
	return embeddings
