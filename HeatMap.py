# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:09:58 2022

@author: anton
"""

import pandas as pd 
import numpy as np 
import matplotlib as mtp
import matplotlib.pyplot as plt 
import seaborn as sns 

path_dir1="G:/Altri computer/Computer_Laboratorio/"
#path_dir1="C:/Users/df_scanM866527/Desktop/"

scan_file="Models_scanchains"
df_scan=pd.read_excel(path_dir1+"AntonioM/" + scan_file + ".xlsx")
df_scan=df_scan.drop("Unnamed: 0",axis=1)

df_scan["estimator"]=df_scan["name_model_radius"].apply(lambda x: x.split("_")[-3])

estimator_list=["LR",'GPR','SVR','KNR',"RF","XGB"]

dataset_dict={}  
for estimator in estimator_list:
        dataset_dict[estimator] = df_scan.loc[(df_scan["estimator"]) == estimator]
        dataset= dataset_dict[estimator]

        MAE_df=pd.DataFrame(data=[dataset["N5_radius"],dataset["NNB_radius"],round(dataset["MAE_test"],2)]).transpose()
        MAE_df_pivoted=MAE_df.pivot(columns="NNB_radius",index="N5_radius",values="MAE_test")
        
        R2_df=pd.DataFrame(data=[dataset["N5_radius"],dataset["NNB_radius"],round(dataset["R2"],2)]).transpose()
        R2_df_pivoted=R2_df.pivot(columns="NNB_radius",index="N5_radius",values="R2")
                
        bwr_reversed = mtp.cm.get_cmap('bwr_r')
        
        f, axs = plt.subplots(1, 2, figsize=(15,5))
        sns.heatmap(data=MAE_df_pivoted, cmap="bwr", cbar=True, cbar_kws={'label': 'MAE',"orientation": "horizontal"},
                       annot=True,fmt=".3g", square=True,vmax=(60),vmin=(20),linewidths=0.3,ax=axs[0]).invert_yaxis()
        
        sns.heatmap(data=R2_df_pivoted, cmap="bwr_r", cbar=True, cbar_kws={'label': 'R2', "orientation": "horizontal"},
                         annot=True, fmt=".3g", square=True,vmax=(1), vmin=(0.4),linewidths=0.3,ax=axs[1]).invert_yaxis()
        
        plt.suptitle(estimator,fontweight ="bold") 
        axs[0].set_title("MAE", fontweight ="bold")
        axs[1].set_title("R2", fontweight ="bold")
        plt.show()   
        
                                  
