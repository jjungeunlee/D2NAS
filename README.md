# [D2NAS: Efficient Neural Architecture Search with Performance Improvement and Model Size Reduction for Diverse Tasks](https://doi.org/10.1109/ACCESS.2024.3434743)

## Introduction
Neural Architecture Search (NAS) has proven valuable in many applications such as computer vision. However, its true potential is unveiled when applied to less-explored domains. In this work, we study NAS for addressing problems in diverse fields where we expect to apply deep neural networks in the real world domains. We introduce D2NAS, Differential and Diverse NAS, leveraging techniques such as Differentiable ARchiTecture Search (DARTS) and Diverse-task Architecture SearcH (DASH) for architecture discovery. Our approach outperforms existing models, including Wide ResNet (WRN) and DASH, when evaluated on NAS-Bench-360 tasks, which include 10 numbers of diverse tasks for 1D, 2D, 2D Dense (tightly interconnected data) tasks. Compared to DASH, D2NAS reduces average error rates by 12.2%, while achieving an 85.1% reduction in average parameters (up to 97.3%) and a 91.3% reduction in Floating Point Operations (FLOPs, up to 99.3%). Therefore, D2NAS enables the creation of lightweight architectures that exhibit superior performance across various tasks, extending its applicability beyond computer vision to include mobile applications.


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
