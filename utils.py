import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
from torch.utils import data

import pickle as pkl
from tqdm import tqdm

import sys
from copy import deepcopy
import os 

# from base_model import Model 

np.random.seed(0)

if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

def to_categorical(y, num_classes):
	# One hot encoding... this may need to be changed to accomadate 
	return np.eye(num_classes, dtype='uint8')[y]

def read_texts(tarfname, dname):
	"""
	FROM LM HW

	Given the location of the data archive file and the name of the
	dataset (one of brown, reuters, or gutenberg), this returns a
	data object containing train, test, and dev data. Each is a list
	of sentences, where each sentence is a sequence of tokens.
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
	train_mem = tar.getmember(dname + ".train.txt")
	train_txt = unicode(tar.extractfile(train_mem).read(), errors='replace')
	test_mem = tar.getmember(dname + ".test.txt")
	test_txt = unicode(tar.extractfile(test_mem).read(), errors='replace')
	dev_mem = tar.getmember(dname + ".dev.txt")
	dev_txt = unicode(tar.extractfile(dev_mem).read(), errors='replace')

	from sklearn.feature_extraction.text import CountVectorizer
	count_vect = CountVectorizer()
	count_vect.fit(train_txt.split("\n"))
	tokenizer = count_vect.build_tokenizer()
	class Data: pass
	data = Data()
	data.train = []
	for s in train_txt.split("\n"):
		toks = tokenizer(s)
		if len(toks) > 0:
			data.train.append(toks)
	data.test = []
	for s in test_txt.split("\n"):
		toks = tokenizer(s)
		if len(toks) > 0:
			data.test.append(toks)
	data.dev = []
	for s in dev_txt.split("\n"):
		toks = tokenizer(s)
		if len(toks) > 0:
			data.dev.append(toks)
	print(dname," read.", "train:", len(data.train), "dev:", len(data.dev), "test:", len(data.test))
	return data

def get_data(DATASET, N_GRAM=20, tknizer=None, save_tokenizer=False):
	"""
	Create a data set that will work for next word prediction task.

	TO DO: This is going to need to return the tokinized sentences from the intended transfer scenario.
	"""

	assert (DATASET == "brown" or DATASET.startswith("reddit"))

	if DATASET == "brown":
		data = read_texts("corpus/corpora.tar.gz", DATASET)
	else:
		data = read_texts("data/{}/{}.tar.gz".format(DATASET, DATASET), DATASET)

	original_data_obj = deepcopy(data)

	# Expand training data by iterating through every sentence 
	# and use just the words seen so far as the training data
	train_sentences = []
	dev_sentences = []
	test_sentences = []
	for s in tqdm(data.train, desc="Constructing training data"):
		for i in range(len(s)):
			train_sentences.append(s[:i+1])

	for s in tqdm(data.dev, desc="Constructing dev data"): 
		for i in range(len(s)):
			dev_sentences.append(s[:i+1])

	for s in tqdm(data.test, desc="Constructing test data"): 
		for i in range(len(s)):
			test_sentences.append(s[:i+1])

	# Tokenize!
	if tknizer is None:
		tknizer = Tokenizer()
		tknizer.fit_on_texts(train_sentences)
	else:
		tkn = pkl.load( open( tknizer, "rb" ) )
		tknizer = tkn['tokenizer']

	if save_tokenizer:
		tkn = {'tokenizer':tknizer}
		pkl.dump( tkn, open( "tokenizers/{}-tokenizer.pkl".format(DATASET), "wb" ) )


	# Pad to the max length of all the occuring sentences
	# Note: don't think we need to calculate max length since TF 
	# does this automatically 

	print("Tokenizing and padding sequences ...")
	# X_train is enumerated version
	X_train = np.array(pad_sequences(tknizer.texts_to_sequences(train_sentences)))
	X_dev = np.array(pad_sequences(tknizer.texts_to_sequences(dev_sentences)))
	X_test = np.array(pad_sequences(tknizer.texts_to_sequences(test_sentences)))

	# X_train is enumerated text
	X_train_orig = np.array(pad_sequences(tknizer.texts_to_sequences(data.train)))
	X_dev_orig = np.array(pad_sequences(tknizer.texts_to_sequences(data.dev)))
	X_test_orig = np.array(pad_sequences(tknizer.texts_to_sequences(data.test)))
	print("Done tokenization and padding.")

	# We have sparse labels here
	X_train, Y_train = X_train[:,-N_GRAM - 1:-1], X_train[:,-1]
	X_dev, Y_dev = X_dev[:,-N_GRAM - 1:-1], X_dev[:, -1]
	X_test, Y_test = X_test[:,-N_GRAM - 1:-1], X_test[:, -1]

	# Package everything together
	data.X_train_orig = X_train_orig
	data.X_dev_orig = X_dev_orig
	data.X_test_orig = X_test_orig
	data.X_train_enumerated = X_train
	data.X_dev_enumerated = X_dev 
	data.X_test_enumerated = X_test 
	data.Y_train_enumerated = Y_train
	data.Y_dev_enumerated = Y_dev 
	data.Y_test_enumerated = Y_test

	return data

def totorch(x, device,grad=False):
	# Pass array to torch
	return torch.Tensor(x).to(device)

def data_generator(data, labels, batch_size):
	for i in range(0, data.shape[0], batch_size):
		yield data[i:i+batch_size], labels[i:i+batch_size]

def load_model(
	model,
	optimizer: torch.optim.Optimizer,
	path: str, 
	train: bool
) -> dict:
	"""
	Loads the model and optimizer state into the given model and 
	optimizer using the path provided. 
	Parameters
	----------
	model: Model 
		The model to load the model state into.
	optimizer: torch.optim.Optimizer
		The optimizer to load the optimizer state into.
	path: str
		The location of the checkpoint.
	train: bool
		Specifies whether to return model in train or eval mode
	"""
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['opt_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	it = checkpoint['it']

	if train:
		model.train()
	else: 
		model.eval()

	return { 'model': model, 'optimizer': optimizer, 'epoch': epoch, 'loss': loss, 'it': it } 

def save_checkpoint(
	epoch: int,
	it: int,
	model_state_dict: dict,
	opt_state_dict: dict,
	loss: float,
	dir_name: str,
	file_name: str
) -> None:
	"""
	Saves the model & optimizer state, as well as the epoch, iteration, 
	and loss. The location of the checkpoint is determined by the directory 
	and file name. 
	"""
	if not os.path.exists(dir_name):
		print("Creating directory", dir_name)
		os.mkdir(dir_name)

	PATH = os.path.join(dir_name, file_name)
	torch.save({
		'epoch': epoch, 
		'it': it,
		'model_state_dict': model_state_dict,
		'opt_state_dict': opt_state_dict,
		'loss': loss 
	}, PATH)



