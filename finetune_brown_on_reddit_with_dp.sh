python3 finetune_brown_on_reddit_with_dp.py --exp_num 1 \
                     --l2_norm_clip 1.0 \
                     --noise_multiplier 0.1 \
                     --minibatch_size 64 \
                     --microbatch_size 10 \
                     --delta 1e-5 \
                     --iterations 12000
