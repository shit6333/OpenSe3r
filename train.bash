torchrun --nproc_per_node=1 train.py \
    --batch_size 1 \
    --interp_v2 \
    --model "AMB3R(metric_scale=True)" \
    --batch_size 1 \
    --accum_iter 4 \
    --epochs 30