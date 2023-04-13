CUDA_VISIBLE_DEVICES=4 python train.py --add_attn --batch_sizes 25 --experiment 1 \
--iterations 800000 --hidden_size 360 --dataset 'mobile' --control 'revenue'