#!/bin/bash

# This script is used to train the model on ALL the experiments & datasets
# It is a wrapper around the train.py script
# This script takes no arguments

models=(
    # "FacebookAI/roberta-base"
    # "FacebookAI/roberta-large"
    "t5-base"
    # "t5-large"
)

data_flags=(
    # "Utt"
    # "Utt_ST"
    # "Utt_DH"
    # "Utt_DA-E"
    # "Utt_ST_DH"
    # "Utt_ST_DA-E"
    # "Utt_DH_DA-E"
    # "Utt_ST_DH_DA-E"
    "DH"
    "DH_ST"
    "DH_DA-E"
    "DH_ST_DA-E"
)

experiments=(
    # "TR-VSA-VSB"
    # "TR-VSB-VSA"
    # "TR-VUA-VUB"
    # "TR-VUB-VUA"
    # "TR-VS-VU"
    # "TR-VU-VS"
    "TR-V-V"
)

for model in "${models[@]}"; do
    for data_flag in "${data_flags[@]}"; do
        for experiment in "${experiments[@]}"; do
            python train.py $model $experiment $data_flag
        done
    done
done

