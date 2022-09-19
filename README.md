# Prediction_Flavoprotein_EM

This repository contains the data and script used for the manuscript "Machine learning for efficient prediction of protein redox potential: the flavoproteins case"
submitted to the Journal of Chemical Information and Modeling.

The dataset of flavoproteins is available at data/dataset.xlsx

The script to run the ML pipeline is available 
* Em predict.py: script for the ML pipeline used to test the performance of the various ML models

* Features Calculator.py: script for automatically extract the features from the PDB files of the flavoproteins contained in data/dataset.xlsx
* SHAP_analysis.ipynb: notebook for SHAP analysis

* HeatMap.py: script to create the table of performance of the various estimator for different combinations of radii r1 and r2

![workflow](https://github.com/CompBtBs/Prediction_Flavoprotein_EM/blob/54c87ac1e69652e538d993009edc52cfc2d44e69/workflow.png)

 
