import torch
from torch import nn
from torch.nn import functional as F

from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import *

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
        self.fc3 = nn.Linear(5000,1000)
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

    def predict_probs(self, x):

        return F.softmax(self.forward(x) / 1, dim=1).to('cpu').data.numpy()

def get_perplexity(model: Model, X_data: np.ndarray, Y_data: np.ndarray, batch_size: int) -> float:
    num_words = 0
    sum_log_probs = 0
    for x_batch, y_batch in data_generator(X_data, Y_data, batch_size):
        x_batch = totorch(x_batch, "cuda:0").float()
        y_batch = totorch(y_batch, "cuda:0").long()

        batch_probs = F.softmax(model(x_batch), dim=1)
        # print(batch_probs)
        # print(batch_probs[np.arange(y_batch.shape[0]), y_batch])
        sum_log_probs += torch.sum(torch.log2(batch_probs[np.arange(y_batch.shape[0]), y_batch])).cpu().data.numpy()
        num_words += y_batch.shape[0]

    return -(1/num_words) * sum_log_probs

def get_next_token_accuracy(model: Model, X_data: np.ndarray, Y_data: np.ndarray, batch_size: int) -> float: 
    acc = 0
    for x_batch, y_batch in data_generator(X_data, Y_data, batch_size):
        x_batch = totorch(x_batch, "cuda:0").float()
        y_batch = totorch(y_batch, "cuda:0").long()

        batch_preds = model.predict(x_batch)

        acc = (acc + np.sum(batch_preds == y_batch.cpu().data.numpy()) / y_batch.shape[0])/2

    return acc 

if __name__ == "__main__":
    N_COMMENTS = "10000" # This is the size of the subset of the Reddit comment dataset we want
    BASE_DSET = "reddit-" + N_COMMENTS
    use_cuda = torch.cuda.is_available()
    print ("Using Cuda: {}".format(use_cuda))
    DEVICE = torch.device("cuda:0" if use_cuda else "cpu")

    tkn = pkl.load( open( "tokenizers/brown-tokenizer.pkl", "rb" ) )
    tknizer = tkn['tokenizer']
    reverse_word_map = dict(map(reversed, tknizer.word_index.items()))

    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return(words)

    batch_size = 64
    TOKENS = 36743

    # LM = Model(X_train.shape[1], Y_train.shape[0])
    # NOTE: we do + 1 here since the padding token is at position 0
    LM = Model(20, TOKENS)
    LM.to(DEVICE)
    # Setup loss
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(LM.parameters(), lr=1e-3)

    # EXAMPLE OF MODEL LOAD
    model_state = load_model(LM, optimizer, "finetune_brown_big_on_reddit_10k/best_model.tar", False)
    LM = model_state['model']
    optimizer = model_state['optimizer']

    sentence = [["Bob", "lives", "close", "to"]]
    token_s = pad_sequences(tknizer.texts_to_sequences(sentence),maxlen=20)



    for _ in range(10):
        preds = LM.predict_probs(totorch(token_s, "cuda:0"))
        pred = np.random.choice(TOKENS,p=preds[0])

        l = list(token_s[0][1:])
        l.append(pred)

        token_s[0] = np.array(l)

    
    print (sequence_to_text(token_s[0]))






