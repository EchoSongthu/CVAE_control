CUDA_VISIBLE_DEVICES=7,6 python train.py --add_attn --batch_sizes 75 --experiment 1 \
--iterations 800000 --hidden_size 360 --dataset 'mobile' --control 'age'