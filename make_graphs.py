import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict

FPATH = 'assessments/'

# FNAMES = ['brown_model_assessed_on_reddit_10k.txt',
#           'reddit_10k_model_assessed_on_reddit_10k.txt',
#           'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k_noise_0.1.txt',
#           'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k_noise_1.1.txt',
#           'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k.txt']

FNAMES = ['brown_big_on_reddit_10k.txt',
          'reddit_10k_big_on_reddit_10k.txt',
          'finetuned_brown_big_on_reddit_10k_assessed_on_reddit_10k_noise_0.1.txt',
          'finetuned_brown_big_on_reddit_10k_assessed_on_reddit_10k_noise_1.1.txt',
          'finetuned_brown_big_on_reddit_10k_assessed_on_reddit_10k.txt']

#FNAMES = ['brown_model_assessed_on_brown.txt']

# Loads the text files of the outputs, processess them so they are usable
def load_data():
    data_dict = defaultdict(str)

    for fname in FNAMES:
        lines = [line.rstrip() for line in open(FPATH + fname)]
        n = len(lines)
        data = []
        for i in range(0, n-2, 2):
            line_dev = lines[i].split()
            line_test = lines[i+1].split()
            data.append((float(line_dev[1]), float(line_dev[3]), float(line_dev[-1]), float(line_test[-1])))

        data_dict[fname] = data[:-1]

    return data_dict

# Function that does the actual plotting
def make_graphs():
    fname_label_dict = {'brown_model_assessed_on_reddit_10k.txt': 'Brown / Reddit_10k',
                        'reddit_10k_model_assessed_on_reddit_10k.txt': 'Reddit_10k / Reddit_10k',
                        'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k_noise_0.1.txt': r'Fine Tuned / Reddit_10k ($\sigma = 0.1$)',
                        'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k_noise_1.1.txt': r'Fine Tuned / Reddit_10k ($\sigma = 1.1$)',
                        'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k.txt': r'Fine Tuned / Reddit_10k ($\sigma = 0$)'
    }

    N_PTS = 25
    data_dict = load_data()


    for fname in FNAMES:
        data = data_dict[fname]

        epochs = [d[0]for d in data][:N_PTS]
        print(epochs)
        iters = [d[1] for d in data][:N_PTS]
        pp_dev = [d[2] for d in data][:N_PTS]
        pp_test = [d[3] for d in data][:N_PTS]

        true_iter = []
        for d in data: 
            if fname == 'finetuned_brown_big_on_reddit_10k_assessed_on_reddit_10k_noise_0.1.txt':
                true_iter.append((d[0] - 1) * 12000 + d[1])
            else: 
                true_iter.append((d[0] - 1) * 10500 + d[1])

        true_iter = true_iter[:N_PTS]
        # true_iter = [(d[0] - 1) * 10500 + d[1] for d in data][:N_PTS]

        if fname == 'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k_noise_0.1.txt':
            linestyle = '--'
        else:
            linestyle = 'solid'
        plt.semilogy(true_iter, pp_dev, label=fname_label_dict[fname], linestyle=linestyle)


    plt.legend()
    plt.xlabel('Training Iteration Count')
    plt.ylabel('Perplexity')
    plt.savefig('plot.png')

make_graphs()
