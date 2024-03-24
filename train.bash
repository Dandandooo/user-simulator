#!/bin/bash

# This script is used to train the model on ALL the experiments & datasets
# It is a wrapper around the train.py script
# This script takes no arguments

models=(
    # "Utt"
    # "Utt_ST"
    "Utt_DH"
    "Utt_DA-E"
    "Utt_ST_DH"
    "Utt_ST_DA-E"
    "Utt_DH_DA-E"
    "Utt_ST_DH_DA-E")
experiments=("TR-VSA-VSB" "TR-VSB-VSA" "TR-VUA-VUB" "TR-VUB-VUA" "TR-VS-VU" "TR-VU-VS" "TR-V-None")

for model in "${models[@]}"; do
    for experiment in "${experiments[@]}"; do
        python train.py $experiment $model
    done
done

