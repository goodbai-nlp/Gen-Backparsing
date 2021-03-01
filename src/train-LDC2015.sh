#!/bin/bash
data_prefix="./workspace/LDC2015-10000-60-5000/LDC2015"
alpha=0.1
beta=0.01
nmt=1.0

adrop=0.1
ldrop=0.4
hdrop=0.5

exp_id=integrated-$adrop-$ldrop-$hdrop-2080Ti
model_dir=./workspace/LDC2015-$alpha-$beta-$nmt-$exp_id/

if [ ! -d "$model_dir" ]; then mkdir -p "$model_dir"; fi
CUDA_VISIBLE_DEVICES=0 nohup python3 -u ./train.py \
                        -data $data_prefix \
                        -save_model $model_dir \
                        -world_size 1 \
                        -gpu_ranks 0 \
                        -save_checkpoint_steps 10000 \
                        -valid_steps 10000 \
                        -report_every 10000 \
                        -keep_checkpoint 10 \
                        -seed 3435 \
                        -train_steps 550000 \
                        -warmup_steps 16000 \
                        --share_decoder_embeddings \
                        -share_embeddings \
                        --position_encoding \
                        --optim adam \
                        -adam_beta1 0.9 \
                        -adam_beta2 0.98 \
                        -decay_method noam \
                        -learning_rate 0.5 \
                        -max_grad_norm 0.0 \
                        -batch_size 2048 \
                        -batch_type tokens \
                        -accum_count 1 \
                        -normalization tokens \
                        -dropout 0.3 \
                        -label_smoothing 0.1 \
                        -max_generator_batches 100 \
                        -param_init 0.0 \
                        -param_init_glorot \
                        -valid_batch_size 8 \
                        -integrated \
                        -integrated_mode cat_all \
                        -a_drop $adrop \
                        -l_drop $ldrop \
                        -h_drop $hdrop \
                        -ratio_alpha $alpha \
                        -ratio_beta $beta \
                        -ratio_nmt $nmt \
                        > LDC2015-joint-integrated-$alpha-$beta-$nmt-$exp_id.log 2>&1 &
