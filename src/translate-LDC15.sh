#!/bin/bash

total_step=500000
start_step=210000
gap=10000

cate=dev
setting=post2
vocab_dir=LDC2015-$setting

# modellist=(0.1-0.0-1.0-integrated-catall-0.1-0.4-0.5)
modellist=(0.1-0.01-1.0-integrated-catall-0.1-0.4-0.5-embnew)

dev=0
for model in ${modellist[@]}
do
tag=LDC2015-$model-$setting
base_dir=/home2/xfbai/mywork/AMR2Text/AMR-Backparsing
train_test_data_dir=$base_dir/data/LDC2015-$setting
model_file=./workspace/LDC2015-$setting-$model/_step_
reference="$train_test_data_dir/${cate}.sent"
output_dir='./workspace/translate-result'
if [ ! -d "$output_dir" ]; then mkdir -p "$output_dir"; fi
hypothesis=$output_dir/${cate}.${tag}
bleu_result="./workspace/LDC2015-$setting-$model/${cate}.${tag}"
#bleu_result="./workspace/LDC2015-4096-10000-$setting-$model/${cate}.${tag}"

if [ ! -d "$bleu_result" ]; then touch "$bleu_result"; fi
for((step=start_step;step<=total_step;step+=gap))
do
CUDA_VISIBLE_DEVICES=$dev python3  ./translate.py -model  $model_file$step.pt \
                                                 -data ./workspace/$vocab_dir/gq-$setting \
                                                 -src        $train_test_data_dir/${cate}.concept.bpe \
                                                 -structure  $train_test_data_dir/${cate}.path  \
                                                 -output     $hypothesis \
                                                 -beam_size 5 \
                                                 -batch_size 25 \
                                                 -share_vocab  \
                                                 -gpu 0 \
                                                 --integrated_node \
                                                 --integrated_edge \

python3 ../eval_utils/back_together.py -input $hypothesis -output $hypothesis
echo "$step," >> $bleu_result
perl ../eval_utils/multi-bleu.perl $reference < $hypothesis >> $bleu_result
done
done
