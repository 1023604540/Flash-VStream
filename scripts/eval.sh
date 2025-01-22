#!/bin/bash

# set up python environment
conda activate vstream1
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
# set important configurations
ngpus=8
gputype=A100

# auto calculate configurations
gpus_list=$(seq -s, 0 $((ngpus - 1)))
date_device="$(date +%m%d)_${ngpus}${gputype}"

echo start eval
# define your openai info here
OPENAIKEY=""
OPENAIBASE="https://api.openai.com"
OPENAITYPE="openai"
OPENAIVERSION="v1"

for dataset in msrvtt
do
    echo start eval ${dataset}
    python -m flash_vstream.eval_video.eval_any_dataset_features \
        --model-path /home/vault/b232dd/b232dd16/Flash-VStream/checkpoints-finetune/vstream-7b-finetune-weighted_kmeans1*8-25*4-25*1-Original/checkpoint-5900 \
        --dataset ${dataset} \
        --num_chunks $ngpus \
        --api_key $OPENAIKEY \
        --api_base $OPENAIBASE \
        --api_type $OPENAITYPE \
        --api_version $OPENAIVERSION \
        >> ${date_device}_vstream-7b-eval-${dataset}.log 2>&1
done


