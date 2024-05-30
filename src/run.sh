#!/bin/bash

DS_LIST=$1 # CIFAR100 SPHERICAL NINAPRO FSD DARCYFLOW PSICOV COSMIC ECG SATELLITE DEEPSEA # MNIST MUSIC

e_id=""
opt=""
if [ "$2" = "dash" ] ; then
    e_id="dash"
    opt="--org_dash"
elif [ "$2" = "common" ] ; then
    e_id="common"
    opt="--common"
elif [ "$2" = "custom" ] ; then
    e_id="custom"
else
    exit
fi

for seed in 0 1 2 ; do
    
    for ds in $DS_LIST ; do
        python3 -W ignore main.py --dataset ${ds} --experiment_id ${e_id} --seed ${seed} ${opt} |& tee ${e_id}/run_${ds}_${seed}.log
    done

done
