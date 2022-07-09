# Prediction_Flavoprotein_EM

This repository contains the data and script used for the manuscript "Machine learning for efficient prediction of protein redox potential: the flavoproteins case"
submitted to the Journal of Chemical Information and Modeling.

The dataset of flavoproteins is available at data/dataset.xlsx

The script to run the ML pipeline is available 
* EM_predict.py: script for the ML pipeline used to test the performance of the various ML models

* Automated_features_extraction.py: script for automatically extract the features from the PDB files of the flavoproteins contained in data/dataset.xlsx
* SHAP_analysis.ipynb: notebook for SHAP analysis

* HeatMap_models_scan.py: script to create the table of performance of the various estimator for different combinations of radii r1 and r2

![intro_paper](https://github.com/CompBtBs/Prediction_Flavoprotein_EM/blob/261b38cad98d52805f6f5f12ac0d069a0229c02e/workflow.png)


