# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 08:53:52 2022

@author: bruno.galuzzi
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu,ks_2samp,ttest_ind

name_models=["LR","GPR","KNR","SVR","RF","XGB"]
df=pd.read_excel("outputs/results_dataset_luglio.xlsx").drop("Unnamed: 0",axis=1).set_index("name_model_radius")

mae_dict=dict()
#dict with mae_list associated for best radius combination for each estimator
for model in name_models:
    df_sub=df[df["estimator"]==model]
    best_radius_comb=df_sub["RMSE"].idxmax()
    mae_list=df["RMSE_list"].loc[best_radius_comb][1:-1].split(",")
    mae_list=[float(x) for x in mae_list]
    mae_dict[model]=mae_list
df_test=pd.DataFrame(index=name_models,columns=name_models)
dict_stats=dict()
for est1 in name_models:
    for est2 in name_models:    
        if est1!=est2:
            dict_stats[(est1+"_"+est2)]=list(mannwhitneyu(mae_dict[est1], 
                                                          mae_dict[est2], 
                                                          alternative="less"))
            df_test.loc[est1,est2]=list(mannwhitneyu(mae_dict[est1], 
                                                          mae_dict[est2], 
                                                          alternative="less"))[1]
#%%


file1="outputs/results_dataset_luglio.xlsx"
file2="data/dataset_input_clean.xlsx"
#%%
df1=pd.read_excel(file2,index_col=0)
names=["LR","SVR","KNR","GPR","RF","XGB"]
df_perf=pd.DataFrame(index=names,columns=["MAE","RMSE","PEARSON","R2","SC"])
dict_models={}
for name in names:
    df_model=df1[df1["estimator"]==name]
    index_min=df_model["MAE_test"].argmin()
    df_model=df_model.iloc[index_min]
    mae_list=df_model["Mae_test_list"]
    mae_list=[float(el) for el in mae_list[1:-2].split(",")]
    dict_models[name]=mae_list

#%%
df2=pd.read_excel(file2)
plt.hist(df2["Em"],bins=20)
plt.grid()
plt.xlabel("Em")
plt.ylabel("Counts")