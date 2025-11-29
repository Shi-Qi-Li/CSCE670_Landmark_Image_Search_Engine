python train.py \
    --dataset_dir /data/hkzhang/GLDv2 \
    --device cuda:1 \
    --hidden_dim 512 \
    --multihead_num 4 \
    --layers_num 4 \
    --lr 1e-4 \
    --batch_size 2048 \
    --num_epochs 180