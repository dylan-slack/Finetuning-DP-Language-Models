python3 train_reddit_10k_model_with_dp.py --exp_num 1 \
                     --l2_norm_clip 1.0 \
                     --noise_multiplier 0.0001 \
                     --minibatch_size 64 \
                     --microbatch_size 1 \
                     --delta 1e-5 \
                     --iterations 12000
