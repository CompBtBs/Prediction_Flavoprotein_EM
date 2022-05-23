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

path_dir1="outputs/"
output_path="outputs/scan/"
scan_file="results_no_5_features_DS_noCovalentLigand.xlsx"
df_scan=pd.read_excel(path_dir1+ scan_file)
df_scan=df_scan.drop("Unnamed: 0",axis=1)

df_scan["estimator"]=df_scan["name_model_radius"].apply(lambda x: x.split("_")[-3])

estimator_list=["LR",'GPR','SVR','KNR',"RF","XGB"]

dataset_dict={}  
for estimator in estimator_list:
        dataset_dict[estimator] = df_scan.loc[(df_scan["estimator"]) == estimator]
        dataset= dataset_dict[estimator]

        MAE_df=pd.DataFrame(data=[dataset["ring_radius"],dataset["bar_radius"],round(dataset["MAE_test"],2)]).transpose()
        MAE_df_pivoted=MAE_df.pivot(columns="bar_radius",index="ring_radius",values="MAE_test")
        
        R2_df=pd.DataFrame(data=[dataset["ring_radius"],dataset["bar_radius"],round(dataset["R2"],2)]).transpose()
        R2_df_pivoted=R2_df.pivot(columns="bar_radius",index="ring_radius",values="R2")
                
        bwr_reversed = mtp.cm.get_cmap('bwr_r')
        
        f= plt.figure(figsize=(25,10))
        
        sns.set_theme(font="Calibri" , font_scale=3.5)
        
        sns.heatmap(data=MAE_df_pivoted, cmap="bwr", cbar=True, cbar_kws={'label': 'MAE',"orientation": "horizontal"},
                       annot=True,fmt=".3g", square=True,vmax=(65),vmin=(30), linewidths=0.3,  annot_kws={"size": 42}, 
                       ).invert_yaxis()
        
        # sns.heatmap(data=R2_df_pivoted, cmap="bwr_r", cbar=True, cbar_kws={'label': 'R2', "orientation": "horizontal"},
        #                  annot=True, fmt=".3g", square=True,vmax=(1), vmin=(0.4), linewidths=0.3, annot_kws={"size": 22},
        #                  ax=axs[1]).invert_yaxis()
        
        f.suptitle(estimator,fontweight ="bold") 
        #axs[0].set_title("MAE", fontweight ="bold")
        #axs[1].set_title("R2", fontweight ="bold")
        plt.show()
        f.savefig(output_path+"plot_"+estimator+"_MAE.svg")
        
                                  
