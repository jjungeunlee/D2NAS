# D2NAS: Efficient Neural Architecture Search with Performance Improvement and Model Size Reduction for Diverse Tasks
## Introduction

## Requirements

To run the code, install the dependencies: 
```bash
#conda create --name d2nas python=3.8
#conda activate d2nas

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install scipy scikit-learn tqdm h5py pandas
pip install graphviz #for DARTS
pip install ml-collections
pip install librosa

git clone https://github.com/mkhodak/relax relax
cd relax && pip install -e .
```

## Preparation NAS-Bench-360 datasets

1. Visit [NAS-Bench-360](https://nb360.ml.cmu.edu) and download dataset into `./src/data`
2. In `./src/data`, run `download.sh`.


## Run D2NAS
Under the `./src` directory, run the following commands:

1. Cell-Based Architecture Search
```bash
Usage : run_pdarts.sh DATASET [OPTION]
Example : run_pdarts.sh CIFAR100 common
          run_pdarts.sh DEEPSEA
```


2. Kernel Patterns Search & Hyperparameter Tuning & Retrain
```bash
Usage : run.sh DATASET [OPTION]
Example : run.sh CIFAR100
          run.sh DEEPSEA custom
          run.sh COSMIC common
```
