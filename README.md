# AI based computational framework for drug development

## Introduction
This repository contains the model for network embedding and scripts to run computational pipeline to prioritize putative drug-targets for a disease of interest, described in our paper [Artificial intelligence based computational framework for drug-target prioritization and inference of novel repositionable drugs for Alzheimer’s disease](https://www.biorxiv.org/content/10.1101/2020.07.17.208116v1).

## Citation
If you use the model and scripts for your study, please cite our article:
Shingo Tsuji, Takeshi Hase, Ayako Yachie, Taiko Nishino, Samik Ghosh, Masataka Kikuchi, Kazuro Shimokawa, Hiroyuki Aburatani, Hiroaki Kitano, Hiroshi Tanaka (2020) Artificial intelligence based computational framework for drug-target prioritization and inference of novel repositionable drugs for Alzheimer’s disease. bioRxiv: 2020.07.17.208116.

## How to use
The process of computational pipeline to prioritize putative drug targets is composed of three steps:

Step 1. Network embedding to extract latent features from protein-protein interaction network obtained from [here](https://www.flyrnai.org/DirectedPPI/directed_human_ppi_file.xls).

Note: we generated a file for adjacency matrix of the network [PIN_data.csv](https://www.dropbox.com/s/hts7q28t5lxge43/PIN_data.csv?dl=0) and used the file for out computational pipeline.

Step 2. Building training data using latent features from step 1 together with a list of known drug targets for a disease of interest.

Step 3. Training a classifier based on Xgboost algorithm using training data from step 2.

### Running the scripts and model for each of the three steps

#### Command to run the script for step 1.

`python network_embedding.py`

network_embedding.py loads optimized deep autoencoder model ([network_embedding_model.hdf5](https://www.dropbox.com/s/rkbfxc8tvdis8xl/network_embedding_model.hdf5?dl=0) is the file for the optimized model) to extract latent features from the protein-interaction network ([PIN_data.csv](https://www.dropbox.com/s/hts7q28t5lxge43/PIN_data.csv?dl=0)). “latent_space.txt” includes latent features for each gene in the network.

#### Command to run the script for step.2

`python building_training_data.py`

The script labels each gene in latent_space.txt to build training data for training classification model. The script load a file (list_known_target.csv) for a list of known drug-target genes (for Alzheimer’s disease) and used the list to label genes. We obtained the list of known targets from [drugbank database](https://www.drugbank.ca/).

#### Command to run the script for step. 3

`python Xgboost_training.py`

The script is to train classifier models and to infer putative drug-targets using the trained classifiers. The script uses training data from step 2 and calculates class probability of potential drug-target class for each gene in training data. The script will generates a file “Prediction_results_for_putative_targets.txt” that contains the class probabilities for genes. The higher value of class probability of drug-target class for a given gene indicates that the gene is more likely to be a potential putative drug-target.
