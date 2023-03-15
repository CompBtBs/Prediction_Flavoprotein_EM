# Prediction Flavoprotein EM

This repository contains the data and script used for the manuscript "*Machine learning for efficient prediction of protein redox potential: the flavoproteins case*" submitted to the Journal of Chemical Information and Modeling.

## Flavoprotein Datset

> The Flavoproteins Dataset is available at `data/dataset.xlsx`. For each flavoprotein used in this work we report:

```
- PDB-ID 
- Organism and Family
- Classification and Cofactor-type
- Technique resolution
- Em, E1, E2
- Reference of the experimental work
```

<br/>

## REPO ORGANIZATION

> All the scripts used to reproduce the work are here reported:

- *Em predict.py*: for the ML pipeline used to test the performance of the various ML models;
- *Features Calculator.py*: features computation of the flavoprotein PDB files reported in data/dataset.xlsx;
- *ML_models.py*: contains all the machine learning models hparams;
- *SHAP_analysis.ipynb*: notebook for SHAP analysis;
- *HeatMap.py*: script to create the table of performance of the various estimator for different combinations of radii r1 and r2;
- *env.yml*: yaml file to create a mamba env from file.

<br/>

## ENVIROMENT

> We suggest to create a mamba env to run the script as follow:

- install mamba
- git clone https://github.com/CompBtBs/Prediction_Flavoprotein_EM.git
- cd Prediction_Flavoprotein_EM
- mamba env create -f env.yml
- install PyBioMed using pip

<br/>

![workflow](https://github.com/CompBtBs/Prediction_Flavoprotein_EM/blob/54c87ac1e69652e538d993009edc52cfc2d44e69/workflow.png)
