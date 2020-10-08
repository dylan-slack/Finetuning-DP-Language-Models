import torch 
import numpy as np 

import os 

from train_brown_model import Model, get_perplexity
from utils import get_data, load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

YSIZE = 36743

def compute_pp(
    Y_train: np.ndarray,
    X_dev: np.ndarray,
    Y_dev: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    file : None
) -> None:
    """
    Computes the dev and test perplexities for all the saved models 
    in the 'pretrained_models' directory. 
    """
    # Instantiate model and optimizer 
    LM = Model(X_dev.shape[1], 37610)
    LM.to(DEVICE)
    optimizer = torch.optim.Adam(LM.parameters(), lr=1e-3)

    # if os.path.exists(file):
    #     os.remove(file)

    # Loop through all saved models and record perplexity
    for epoch in range(2, 3):
        for it in range(2500, 12500, 500):
            print(epoch, it)

            model_state = load_model(LM, optimizer, "reddit_10k_big/epoch_{}_it_{}.tar".format(epoch, it), False)
            LM = model_state['model']

            dev_pp = get_perplexity(LM, X_dev, Y_dev, 64)
            test_pp = get_perplexity(LM, X_test, Y_test, 64)
            with open(file, 'a+') as f:  
                f.write("Epoch {} It {} | Dev PP = {}\n".format(epoch, it, dev_pp))
                f.write("Epoch {} It {} | Test PP = {}\n".format(epoch, it, test_pp))

    model_state = load_model(LM, optimizer, "reddit_10k_big/best_model.tar", False)
    LM = model_state['model']

    dev_pp = get_perplexity(LM, X_dev, Y_dev, 64)
    test_pp = get_perplexity(LM, X_test, Y_test, 64)
    with open(file, 'a+') as f:  
        f.write("Best Model Dev PP = {}\n".format(dev_pp))
        f.write("Best Model Test PP = {}\n".format(test_pp))

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print("Using Cuda: {}".format(use_cuda))
    DEVICE = torch.device("cuda:0" if use_cuda else "cpu")

    # load data 
    BASE_DSET = "reddit-10000"
    data_dict = get_data(BASE_DSET, tknizer="tokenizers/reddit-10000-tokenizer.pkl")

    # get relevant data 
    X_train, Y_train = data_dict.X_train_enumerated, data_dict.Y_train_enumerated
    X_dev, Y_dev = data_dict.X_dev_enumerated, data_dict.Y_dev_enumerated
    X_test, Y_test = data_dict.X_test_enumerated, data_dict.Y_test_enumerated

    # compute perplexities for dev and test
    compute_pp(Y_train, X_dev, Y_dev, X_test, Y_test, "assessments/reddit_10k_big_on_reddit_10k.txt")
