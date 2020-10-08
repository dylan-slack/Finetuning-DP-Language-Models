import os
import math
root = "./assessments"

BIGMODELS = True

FNAMES = ['finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k_noise_0.1.txt',
          'finetuned_brown_mode_on_reddit_10k_assessed_on_reddit_10k_noise_1.1.txt']

BIGFNAMES = ['finetuned_brown_big_on_reddit_10k_assessed_on_reddit_10k_noise_0.1.txt',
          'finetuned_brown_big_on_reddit_10k_assessed_on_reddit_10k_noise_1.1.txt']

sigmas = [0.1, 1.1]

def get_perplexity(fname):
	with open(os.path.join(root,fname), 'r') as f:
		data = f.readlines()
		best_perplexity = round(float(data[-1].split()[-1]),2)		
		return best_perplexity

def epsilon_refined_accountant(sigma, delta, q, T):
    return q/sigma * (T * math.log(1./ delta)) ** 0.5

for i, file in enumerate(zip(FNAMES, BIGFNAMES)):

	smallf = file[0]
	bigf = file[1]

	best_small = get_perplexity(smallf)
	best_big = get_perplexity(bigf)

	cur_sigma = sigmas[i]

	q = 0.01  # Mini-batch ratio
	T = 1e5
	delta = 1e-5

	epsilon = epsilon_refined_accountant(cur_sigma, delta, q, T)
	

	print ('----')
	print ("Big: epsilon",epsilon,"perp:",best_big)
	print ("Small: epsilon",epsilon,"perp:",best_small)






