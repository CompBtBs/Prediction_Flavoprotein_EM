# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:25:21 2022

@author: anton
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

name_models=["LR","GPR","KNR","SVR","RF","XGB"]
df=pd.read_excel("outputs/results.xlsx").drop("Unnamed: 0",axis=1).set_index("name_model_radius")

mae_dict=dict()
#dict with mae_list associated for best radius combination for each estimator
for model in name_models:
    df_sub=df[df["estimator"]==model]
    best_radius_comb=df_sub["MAE_test"].idxmin()
    mae_list=df["Mae_test_list"].loc[best_radius_comb][1:-1].split(",")
    mae_list=[float(x) for x in mae_list]
    mae_dict[model]=mae_list

dict_stats=dict()
for est1 in name_models:
    for est2 in name_models:    
        if est1!=est2:
            dict_stats[(est1+"_"+est2)]=list(mannwhitneyu(mae_dict[est1], 
                                                          mae_dict[est2], 
                                                          alternative="less"))
