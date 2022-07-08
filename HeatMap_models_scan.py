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
output_path="figure/scan/"
scan_file="results_dataset_luglio.xlsx"
df_scan=pd.read_excel(path_dir1+ scan_file)
if "Unnamed: 0" in df_scan.columns:
    df_scan=df_scan.drop("Unnamed: 0",axis=1)

df_scan["estimator"]=df_scan["name_model_radius"].apply(lambda x: x.split("_")[-3])
estimator_list=["LR",'GPR','SVR','KNR',"RF","XGB"]

bwr_reversed = mtp.cm.get_cmap('bwr_r')
sns.set_theme(font="Calibri" , font_scale=2.8)
f, axs=plt.subplots(3, 2,figsize=(40,30), gridspec_kw={"hspace":0.4, "wspace":0.005})

cbar_ax = f.add_axes([.3, 0, .41, .03])

dataset_dict={"LR":[axs[0,0],[],[]],'GPR':[axs[0,1],[]],'SVR':[axs[1,0],[]],'KNR':[axs[1,1],[]],"RF":[axs[2,0],[]],"XGB":[axs[2,1],[]]}  
for estimator in estimator_list:
        
        dataset_dict[estimator][1].append(df_scan.loc[(df_scan["estimator"]) == estimator])
        dataset= dataset_dict[estimator][1][0]
        dataset.rename(columns={"bar_radius": "r1", "ring_radius":"r2"},inplace = True)

        MAE_df=pd.DataFrame(data=[dataset["r2"],dataset["r1"],round(dataset["MAE_test"],2)]).transpose()
        MAE_df_pivoted=MAE_df.pivot(columns="r1",index="r2",values="MAE_test")
        
        R2_df=pd.DataFrame(data=[dataset["r2"],dataset["r1"],round(dataset["R2"],2)]).transpose()
        R2_df_pivoted=R2_df.pivot(columns="r1",index="r2",values="R2")
        
        Spearman_df=pd.DataFrame(data=[dataset["r2"],dataset["r1"],round(dataset["Spearman"],2)]).transpose()
        Spearman_df_pivoted=Spearman_df.pivot(columns="r1",index="r2",values="Spearman")
                
        if estimator=="XGB":
            cbar=True
        else:
            cbar=False
            
        # sns.heatmap(data=MAE_df_pivoted, cmap="bwr", cbar=cbar, cbar_ax=None if estimator!="XGB" else cbar_ax,               
        #             cbar_kws={'label': 'MAE',"orientation": "horizontal"},
        #                 annot=True,fmt=".3g", square=True,vmax=(70),vmin=(30), linewidths=0.3,  annot_kws={"size": 42},
        #                 ax=dataset_dict[estimator][0]).invert_yaxis()
        
        sns.heatmap(data=R2_df_pivoted, cmap="bwr_r", cbar=cbar, cbar_ax=None if estimator!="XGB" else cbar_ax,
                        cbar_kws={'label': 'R2',"orientation": "horizontal"},
                          annot=True, fmt=".3g", square=True, vmax=(0.9), vmin=(0.4), linewidths=0.3, annot_kws={"size": 42},
                          ax=dataset_dict[estimator][0]).invert_yaxis()
        
        dataset_dict[estimator][0].set_title(estimator,fontweight ="bold",fontsize=42)
        
       # dataset_dict[estimator][0].set(xlabel=None,ylabel=None)
        
        # if estimator in ["XGB","GPR","KNR"]:
        #     dataset_dict[estimator][0].get_yaxis().set_visible(False)
        
        # if estimator in ["GPR","KNR","SVR","LR"]:
        #     dataset_dict[estimator][0].get_xaxis().set_visible(False)
        
#f.colorbar(im,shrink=0.8,ax=axs.ravel().tolist(),location="bottom")
plt.show()
f.savefig(output_path+"heatmap_R2.svg")
        
        
        
        
                                  
