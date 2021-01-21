#!/bin/bash
base_dir=/home2/xfbai/mywork/AMR2Text/AMR-Backparsing/data
model=post2
train_test_data_dir=$base_dir/LDC2015-$model

data_dir="./workspace/LDC2015-10000-60-5000-${model}/"
if [ ! -d "$data_dir" ]; then mkdir -p "$data_dir"; fi
data_prefix="$data_dir/LDC2015-$model"

python3 ./preprocess.py -train_src $train_test_data_dir/train.concept.bpe \
                        -train_tgt $train_test_data_dir/train.tok.sent.bpe \
                        -train_structure  $train_test_data_dir/train.path  \
                        -train_mask $train_test_data_dir/train.rel \
                        -train_relation_lst $train_test_data_dir/train.rel.lst \
                        -train_relation_mat $train_test_data_dir/train.rel.mat \
                        -train_align $train_test_data_dir/train.align.matrix \
                        -valid_src $train_test_data_dir/dev.concept.bpe  \
                        -valid_tgt $train_test_data_dir/dev.tok.sent.bpe \
                        -valid_structure $train_test_data_dir/dev.path   \
                        -valid_mask $train_test_data_dir/dev.rel   \
                        -valid_relation_lst $train_test_data_dir/dev.rel.lst  \
                        -valid_relation_mat $train_test_data_dir/dev.rel.mat \
                        -valid_align $train_test_data_dir/dev.align.matrix \
                        -save_data $data_prefix \
                        -src_vocab_size 10000  \
                        -tgt_vocab_size 10000 \
                        -structure_vocab_size 5000 \
                        -relation_vocab_size 60 \
                        -src_seq_length 10000 \
                        -tgt_seq_length 10000 \
                        -share_vocab

