#!/bin/bash

DS_LIST=$1 # CIFAR100 SPHERICAL NINAPRO FSD DARCYFLOW PSICOV COSMIC ECG SATELLITE DEEPSEA # MNIST MUSIC

opt=""
if [ "$2" = "common" ] ; then
    opt="--common"
fi

for seed in 0 1 2; do
    
    for ds in $DS_LIST ; do
        python train_search.py \
               --dataset ${ds} \
               --tmp_data_dir ./tmp_${ds} \
               --save ./log \
               --seed ${seed} \
               ${opt}
    done

done