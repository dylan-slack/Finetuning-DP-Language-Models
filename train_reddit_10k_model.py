import torch
from torch import nn
from torch.nn import functional as F

from utils import *

YSIZE = 36743

####
## Train a feedforward neural network to do next token prediction
####

class Model(nn.Module):
	# We can talk about the architecture later on but keeping it simple for now
	def __init__(self, dim_in, dim_out):
		super(Model, self).__init__()
		self.fc1 = nn.Linear(dim_in, 10000)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(10000, 5000)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(5000, 1000)
		self.relu3 = nn.ReLU()
		self.logits = nn.Linear(1000, dim_out)

	def forward(self, x):
		l1 = self.relu1(self.fc1(x))
		l2 = self.relu2(self.fc2(l1))
		l3 = self.relu3(self.fc3(l2))
		return self.logits(l3)

	def predict(self, x):
		if not isinstance(x, torch.Tensor):
			x = totorch(x, 'cpu').float()

		return torch.argmax(F.softmax(self.forward(x), dim=1), dim=1).to('cpu').data.numpy()

def get_perplexity(model: Model, X_data: np.ndarray, Y_data: np.ndarray, batch_size: int) -> float:
	num_words = 0
	sum_log_probs = 0
	for x_batch, y_batch in data_generator(X_data, Y_data, batch_size):

		adj_x_batch = []
		adj_y_batch = []

		for j in range(len(y_batch)):
			if y_batch[j] == 0: 
				continue
			else:
				adj_x_batch.append(x_batch[j])
				adj_y_batch.append(y_batch[j])

		x_batch = np.array(adj_x_batch)
		y_batch = np.array(adj_y_batch)

		x_batch = totorch(x_batch, "cuda:0").float()
		y_batch = totorch(y_batch, "cuda:0").long()

		batch_probs = F.softmax(model(x_batch), dim=1)

		p_x = batch_probs[np.arange(y_batch.shape[0]), y_batch]

		sum_log_probs += torch.sum(torch.log2(p_x)).cpu().data.numpy()
		num_words += y_batch.shape[0]

	pp = -(1/num_words) * sum_log_probs
	return 2 ** pp

def get_next_token_accuracy(model: Model, X_data: np.ndarray, Y_data: np.ndarray, batch_size: int) -> float: 
	acc = 0
	for x_batch, y_batch in data_generator(X_data, Y_data, batch_size):
		x_batch = totorch(x_batch, "cuda:0").float()
		y_batch = totorch(y_batch, "cuda:0").long()

		batch_preds = model.predict(x_batch)

		acc = (acc + np.sum(batch_preds == y_batch.cpu().data.numpy()) / y_batch.shape[0])/2

	return acc 

if __name__ == "__main__":
	# Get the data using the utils scrips
	# NOTE: get_data is going to need to be updated to also return the tokenized transfer scenario
	# using the tokenizer applied to Brown.
	BASE_DSET = "reddit-10000"
	use_cuda = torch.cuda.is_available()
	print ("Using Cuda: {}".format(use_cuda))
	DEVICE = torch.device("cuda:0" if use_cuda else "cpu")
	data_dict = get_data(BASE_DSET, tknizer="tokenizers/reddit-10000-tokenizer.pkl")

	batch_size = 64

	# Get data 
	X_train, Y_train = data_dict.X_train_enumerated, data_dict.Y_train_enumerated
	X_dev, Y_dev = data_dict.X_dev_enumerated, data_dict.Y_dev_enumerated
	X_test, Y_test = data_dict.X_test_enumerated, data_dict.Y_test_enumerated
	print("shape", Y_test.shape)
	# LM = Model(X_train.shape[1], Y_train.shape[0])
	# NOTE: we do + 1 here since the padding token is at position 0
	print(np.max(Y_test) + 1)
	LM = Model(X_train.shape[1], np.max(Y_train) + 1)
	LM.to(DEVICE)
	# Setup loss
	loss_f = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(LM.parameters(), lr=1e-3)

	# EXAMPLE OF MODEL LOAD
	model_state = load_model(LM, optimizer, "reddit_10k_big/epoch_2_it_2000.tar", False)
	LM = model_state['model']
	optimizer = model_state['optimizer']

	for E in range(1, 10):

		# Store losses
		it = 2000
		losses = []
		acc = []
		best_model = {
			'epoch': E + 1,
			'it': it,
			'model_state_dict': LM.state_dict(),
			'opt_state_dict': optimizer.state_dict(),
			'loss': 1e9
		}
		for x_batch, y_batch in tqdm(data_generator(X_train, Y_train, batch_size), desc="Iterating over epoch {}".format(E + 1)):

			# Pass batch to device
			x_batch = totorch(x_batch, DEVICE).float()
			y_batch = totorch(y_batch, DEVICE).long()

			loss = loss_f(LM(x_batch), y_batch)

			batch_preds = LM.predict(x_batch)
			acc.append(np.sum(batch_preds == y_batch.cpu().data.numpy()) / y_batch.shape[0])

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			l = loss.to('cpu').data.numpy()
			losses.append(l)
			# if l < best_model['loss']:
			# 	best_model['loss'] = l
			# 	best_model['epoch'] = E + 1
			# 	best_model['it'] = it
			# 	best_model['model_state_dict'] = LM.state_dict()
			# 	best_model['opt_state_dict'] = optimizer.state_dict()
			# 	save_checkpoint(**best_model, dir_name="reddit_10k_big", file_name="best_model.tar")

			it += 1 

			if it % 500 == 0:
				print ("Epoch {} | Iter {} | Avg Loss On Epoch {} | Avg Next token Acc On Epoch {}".format(E + 1, it, round(np.mean(losses),4), round(np.mean(acc),4)))
				print ("Dev Accuracy", get_next_token_accuracy(LM, X_dev, Y_dev, batch_size))
				print ("Test Accuracy", get_next_token_accuracy(LM, X_test, Y_test, batch_size))
				print ("Dev Perplexity", get_perplexity(LM, X_dev, Y_dev, 64))
				print ("Test Perplexity", get_perplexity(LM, X_test, Y_test, 64))
				# save checkpoint
				save_checkpoint(E + 1, it, LM.state_dict(), optimizer.state_dict(), l, "reddit_10k_big", "epoch_{}_it_{}.tar".format(E + 1, it))








