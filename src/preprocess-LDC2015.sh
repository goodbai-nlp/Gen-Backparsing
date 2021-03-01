#!/bin/bash
BASE=$(dirname $(pwd))
cate=LDC2015
cate=LDC2017
train_test_data_dir=$BASE/data/$cate

data_dir="./workspace/$cate-10000-60-5000/"
if [ ! -d "$data_dir" ]; then mkdir -p "$data_dir"; fi
data_prefix="$data_dir/$cate"

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
