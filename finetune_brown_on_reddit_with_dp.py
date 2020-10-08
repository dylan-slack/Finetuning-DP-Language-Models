import torch
from torch import nn
from torch.utils.data import TensorDataset
import numpy as np
from pyvacy import optim, analysis, sampling

import argparse 
import os 

from train_brown_model import Model, get_next_token_accuracy
from utils import get_data, load_model, totorch, save_checkpoint

class DPFineTuner:
    def __init__(
        self, 
        model, 
        train_dataset, 
        args, 
        device, 
        X_dev, 
        Y_dev,
        X_test,
        Y_test
    ):
        self.train_dataset = train_dataset
        self.model = model
        self.loss_f = nn.CrossEntropyLoss()
        self.device = device

        self.X_dev = X_dev
        self.Y_dev = Y_dev
        self.X_test = X_test
        self.Y_test = Y_test

        self.batch_size = 64

        self.args = args

        # init record files
        self.record_dir = "dp_experiments_big"
        if not os.path.exists(self.record_dir):
            print("Making directory:", self.record_dir)
            os.mkdir(self.record_dir)

        self.exp_dir = os.path.join(self.record_dir, "experiment_{}".format(args.exp_num))
        if not os.path.exists(self.exp_dir):
            print("Making directory:", self.exp_dir)
            os.mkdir(self.exp_dir)

        self.exp_desc = os.path.join(self.exp_dir, "exp_desc.txt")
        self.training_acc_file = os.path.join(self.exp_dir, "training_acc.txt")
        self.dev_acc_file = os.path.join(self.exp_dir, "dev_acc.txt")
        self.test_acc_file = os.path.join(self.exp_dir, "test_acc.txt")
        
        files = [
            self.exp_desc,
            self.training_acc_file,
            self.dev_acc_file, 
            self.test_acc_file
        ]

        # Refresh files 
        for f in files: 
            if os.path.exists(f):
                os.remove(f)

        # TODO: write to experiment description file with hyperparameters

        self.dp_optim_params = {
            # An upper bound on the L2 norm of each gradient update.
            # A good rule of thumb is to use the median of the L2 norms observed
            # throughout a non-private training loop.
            'l2_norm_clip': args.l2_norm_clip,
            # A coefficient used to scale the standard deviation of the noise applied to gradients.
            'noise_multiplier': args.noise_multiplier,
            # Each example is given probability of being selected with minibatch_size / N.
            # Hence this value is only the expected size of each minibatch, not the actual. 
            'minibatch_size': args.minibatch_size,
            # Each minibatch is partitioned into distinct groups of this size.
            # The smaller this value, the less noise that needs to be applied to achieve
            # the same privacy, and likely faster convergence. Although this will increase the runtime.
            'microbatch_size': args.microbatch_size
        }

        self.epsilon_params = {
            'N': len(train_dataset),
            # A coefficient used to scale the standard deviation of the noise applied to gradients.
            'noise_multiplier': args.noise_multiplier,
            # Each example is given probability of being selected with minibatch_size / N.
            # Hence this value is only the expected size of each minibatch, not the actual. 
            'batch_size': args.minibatch_size,
            # The usual privacy parameter for (ε,δ)-Differential Privacy.
            # A generic selection for this value is 1/(N^1.1), but it's very application dependent.
            'delta': args.delta,
            # The number of minibatches to process in the training loop.
            'iterations': args.iterations
        }

        self.sampler_params = {
            # Each example is given probability of being selected with minibatch_size / N.
            # Hence this value is only the expected size of each minibatch, not the actual. 
            'minibatch_size': args.minibatch_size,
            # Each minibatch is partitioned into distinct groups of this size.
            # The smaller this value, the less noise that needs to be applied to achieve
            # the same privacy, and likely faster convergence. Although this will increase the runtime.
            'microbatch_size': args.microbatch_size,
            # The number of minibatches to process in the training loop.
            'iterations': args.iterations
        }

    def fine_tune(self):
        optimizer = optim.DPSGD(params=self.model.parameters(), **self.dp_optim_params, lr=1e-3) 
        epsilon = analysis.epsilon(**self.epsilon_params)
        print("epsilon", epsilon)

        model_state = load_model(self.model, optimizer, "finetuned_brown_big_on_reddit_10k_0.1/epoch_2_it_3500.tar", False)
        self.model = model_state['model']
        optimizer = model_state['optimizer']


        for E in range(1, 5):
            # Store losses
            it = 3500
            losses = []
            acc = []
            best_model = {
                'epoch': E + 1,
                'it': it,
                'model_state_dict': self.model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'loss': 1e9
            }

            minibatch_loader, microbatch_loader = sampling.get_data_loaders(**self.sampler_params)
            for X_minibatch, y_minibatch in minibatch_loader(self.train_dataset):
                optimizer.zero_grad()
                for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                    X_microbatch = X_microbatch.float().to(self.device)
                    y_microbatch = y_microbatch.long().to(self.device)

                    optimizer.zero_microbatch_grad()
                    loss = self.loss_f(self.model(X_microbatch), y_microbatch)
                    loss.backward()
                    optimizer.microbatch_step()

                l = loss.to('cpu').data.numpy()
                losses.append(l)
                # if l < best_model['loss']:
                #     best_model['loss'] = l
                #     best_model['epoch'] = E + 1
                #     best_model['it'] = it
                #     best_model['model_state_dict'] = self.model.state_dict()
                #     best_model['opt_state_dict'] = optimizer.state_dict()
                #     save_checkpoint(**best_model, dir_name="finetuned_brown_big_on_reddit_10k_" + str(args.noise_multiplier), file_name="best_model.tar")

                it += 1 

                if it % 500 == 0:
                    avg_loss = round(np.mean(losses),4)
                    # Write to file
                    with open(self.training_acc_file, "a") as f:
                        f.write("Epoch {} | Iter {} | Avg Loss On Epoch {} | Avg Next token Acc On Epoch {}\n".format(E + 1, it, avg_loss, round(np.mean(acc),4)))
                    with open(self.dev_acc_file, "a") as f:
                        f.write("Epoch {} | Iter {} | Avg Loss On Epoch {} | Avg Next token Acc On Epoch {}\n".format(E + 1, it, avg_loss, get_next_token_accuracy(self.model, self.X_dev, self.Y_dev, self.batch_size)))
                    with open(self.test_acc_file, "a") as f:
                        f.write("Epoch {} | Iter {} | Avg Loss On Epoch {} | Avg Next token Acc On Epoch {}\n".format(E + 1, it, avg_loss, get_next_token_accuracy(self.model, self.X_test, self.Y_test, self.batch_size)))
                    # save checkpoint
                    save_checkpoint(E + 1, it, self.model.state_dict(), optimizer.state_dict(), l, "finetuned_brown_big_on_reddit_10k_" + str(args.noise_multiplier), "epoch_{}_it_{}.tar".format(E + 1, it))

                batch_preds = self.model.predict(X_minibatch.float().to(self.device))
                acc.append(np.sum(batch_preds == y_minibatch.cpu().data.numpy()) / y_minibatch.shape[0])
                optimizer.step()

def main(device, args):
    # Load the reddit train, dev, and test data
    data_dict = get_data("reddit-10000", tknizer="tokenizers/brown-tokenizer.pkl")

    X_train, Y_train = data_dict.X_train_enumerated, data_dict.Y_train_enumerated
    X_dev, Y_dev = data_dict.X_dev_enumerated, data_dict.Y_dev_enumerated
    X_test, Y_test = data_dict.X_test_enumerated, data_dict.Y_test_enumerated

    LM = Model(X_train.shape[1], 36743)
    LM.to(device)
    # Don't need this optimizer here, but the load function requires it
    optimizer = torch.optim.Adam(LM.parameters(), lr=1e-3)
    model_state = load_model(LM, optimizer, "brown_pretrained_big/best_model.tar", train=False)
    LM = model_state['model']

    train_dataset = [(x_train, y_train) for x_train, y_train in zip(X_train, Y_train)]
    # print(train_dataset)
    dp_finetuner = DPFineTuner(
        LM, 
        train_dataset, 
        args, 
        device,
        X_dev,
        Y_dev,
        X_test,
        Y_test
    )
    dp_finetuner.fine_tune()

def parse_args():
    parser = argparse.ArgumentParser(description='Take DP parameters.')
    parser.add_argument('--exp_num', type=int, help='Set value for experiment number')
    parser.add_argument('--l2_norm_clip', type=float, help='Set value for clipping')
    parser.add_argument('--noise_multiplier', type=float, help='Set value for noise multiplier')
    parser.add_argument('--minibatch_size', type=int, help='Set value for mini batch')
    parser.add_argument('--microbatch_size', type=int, help='Set value for micro batch')
    parser.add_argument('--delta', type=float, help='Set value for delta')
    parser.add_argument('--iterations', type=int, help='Set value for iterations')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print ("Using Cuda: {}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    args = parse_args()
    main(device, args)