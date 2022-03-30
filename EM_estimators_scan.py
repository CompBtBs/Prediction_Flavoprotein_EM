# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:18:52 2022

@author: anton
"""

### Importo le librerie

# In[]:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import sys
from statistics import median
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from skopt.space import Real, Categorical, Integer
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF,ConstantKernel


# In[2]:

import warnings
warnings.filterwarnings('ignore')

# In[5]:

path_dir1="G:/Altri computer/Computer_Laboratorio/"
#path_dir1="C:/Users/AM866527/Desktop/"


# In[ciclo_per_più_dataset]:

models_scan=pd.DataFrame(columns=["name_model_raggio","selector","NNB_radius","N5_radius","MAE_train","sd_train","MAE_test",
                                  "sd_test","mean_error_kfold","sd_error_kfold","R2","Pearson","Spearman","n°_selected_features",
                                  "selected_features","Mae_test_list","R2_test_list","Error_Kfold_list"])

covariation=0.99
list_NNB_radius=np.arange(8,17)     
list_N5_radius=np.arange(3,6)
num_cv=3 #cross validation for features selector
num_rep=10 #replicates for features selector
CV=5 #cross validation for hyperparameters tuning
opt="neg_mean_absolute_error" #evaluation metric for hyperparameters tuning
selector="ElasticNetCV" 
sel=30 #coefficients cut-off for features selection
split_tuning=50 #number of split for hyperparameters tuning
test_size=0.33 #train test split 
file_input_name ="proteins" #or proteins 
models_dict={"LR":[LinearRegression(),{}],
             'GPR':[GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0)),                    
                    {
                    'regressor__alpha':np.logspace(-2, 2, 5),
                    'regressor__kernel__k1__constant_value': np.logspace(-2, 2, 5),
                    'regressor__kernel__k2__length_scale': np.logspace(-2, 2, 5)
                    }],
             'SVR':[SVR(RBF()),                   
                    {
                    'regressor__C': np.logspace(-3,3,7),
                    'regressor__gamma': np.logspace(-3,3,7)
                    #'regressor__epsilon': np.logspace(-1,0,2)
                    }],
             'KNR':[KNeighborsRegressor(),
                    {           
                    'regressor__n_neighbors': [2, 3, 4, 5, 6, 7],
                    'regressor__metric': ['euclidean','manhattan'],
                    'regressor__weights':  ["uniform", "distance"]
                    }],
             "RF":[RandomForestRegressor(),
                   {
                    'regressor__n_estimators': [100,150,200],
                    'regressor__max_features': ['auto', 'sqrt','log2'],
                    'regressor__max_depth': [3, 4, 5]
                    }],
             "XGB":[XGBRegressor(),
                    {
                    "regressor__learning_rate" : [0.01,0.1,0.2,0.4],
                    "regressor__max_depth" : [3,4,5],
                    "regressor__min_child_weight" : [1,5,10],
                    "regressor__n_estimators" : [100,150,200]
                    }]
             } 
name_models=models_dict.keys()#["GPR"]




###############################################################################################
# In[]:

index_line=0 #index riga for model scan dataset 

for NNB_radius in list_NNB_radius: 
    for ring_radius in list_N5_radius:
            
#%%
            file_name="dataset_"+file_input_name+"_"+str(NNB_radius)+"_"+str(ring_radius)+".xlsx"
            print("dataset_"+file_input_name+"_"+str(NNB_radius)+"_"+str(ring_radius))
            # Upload dataset
            #file_name="database_chains_8_4.xlsx"
            df_pm=pd.read_excel(path_dir1+"AntonioM/Dataset_finali/"+file_name,sheet_name="Sheet1",index_col=0)
            df_pm=df_pm.reset_index()
            df_pm=df_pm.iloc[:,1:]
            if "Protein_name" in df_pm.columns:
                df_pm=df_pm.drop(["Protein_name"],axis=1) 
            
            columns_to_keep_=[el for el in df_pm.columns if "%" not in el and "_mean" not in el]
            df_pm=df_pm.loc[:,columns_to_keep_]
            
        
# In[]:
# preprocessing

            columns_to_remove=[el for el in df_pm.columns[0:-1] if df_pm[el].std()==0]
            
            df_pm.drop(columns_to_remove,axis=1,inplace=True)
            
            Df_corr=df_pm.corr().abs()
            upper_tri = Df_corr.where(np.triu(np.ones(Df_corr.shape),k=1).astype(np.bool))
            
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= covariation)]
            
            df_pm=df_pm.drop(columns=to_drop,axis=1)
            # Restituisce in output il dataset per fare Machine Learning
                        
            # In[]:
            # Input/output

            X=df_pm.iloc[:,1:].values
            y=df_pm.iloc[:,0].values
            

            # In[]:          
            print("start features selection")
            
            scores_MAE_selector=np.zeros((num_rep,num_cv))
            scores_R2_selector=np.zeros((num_rep,num_cv))
            scores_Pearson_selector=np.zeros((num_rep,num_cv))
            coefsElasticNet=list()
            
            for j in range(num_rep):
                #print(j)
                cv = KFold(num_cv,shuffle=True,random_state=24+j)
                k=0
                for train_index, test_index in cv.split(X):
                    X_train2, X_test2 = X[train_index,:], X[test_index,:]
                    y_train2, y_test2 = y[train_index], y[test_index]
                    
                    if selector == "ElasticNetCV":
                        modelL = Pipeline([
                                ('scaler', StandardScaler()),
                                ('regressor', ElasticNetCV(max_iter=1000))
                                ])  
                    else:
                        
                        modelL = Pipeline([
                                ('scaler', StandardScaler()),
                                ('regressor', LassoCV(max_iter=100))
                                ])  
            
                    modelL.fit(X_train2, y_train2)         
                     
                    scores_MAE_selector[j][k]=mean_absolute_error(y_test2,modelL.predict(X_test2))
                    scores_R2_selector[j][k]=r2_score(y_test2,modelL.predict(X_test2))
                    scores_Pearson_selector[j][k]=pearsonr(y_test2,modelL.predict(X_test2))[0]
                    values=results=modelL.get_params()['regressor'].coef_
            
                    coefsElasticNet.append(values)
                    k=k+1
            
            
            # In[]:
            C=np.sum(np.array(np.abs(coefsElasticNet)>1),axis=0)

            # In[]:
            
            selected_features=[]
            for el,el2 in zip(df_pm.columns[1:],C):
                if el2>=sel:
                    selected_features.append(el)
            
            print("features selection completed")
            print("number of selected features:",len(selected_features))
            # In[]:
                
            X2=X[:,C>=sel]
            
            # In[]:
            # Hyperparameter Tuning
            print( "hyperparameter tuning")
            # In[]:

            DATA_dict={name_model:{"mae_train":[],"mae_test":[],"R2":[],"Pearson":[],"Spearman":[],"EOKfold":[]} for name_model in name_models}

            
            for i in range(split_tuning):
                print("Split_train_test:",i)
                X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=test_size, random_state=i)
                
                for estimator in name_models:
                    if estimator== "LR":
                        
                        optEstimator= models_dict[estimator][0]
                        optEstimator.fit(X_train, y_train)
                        DATA_dict[estimator]["EOKfold"].append(mean_absolute_error(optEstimator.predict(X_train), y_train))
                        
                    else:
                        pipe = Pipeline([('scaler', StandardScaler()),
                                         ('regressor', models_dict[estimator][0])
                                        ])
                        
                        optEstimator = GridSearchCV(pipe, models_dict[estimator][1],
                                                    scoring=opt,cv=CV
                                                    )
                                                    
                        _= optEstimator.fit(X_train, y_train)
                        
                        DATA_dict[estimator]["EOKfold"].append(np.abs(optEstimator.best_score_))
                        
                    DATA_dict[estimator]["mae_train"].append(mean_absolute_error(optEstimator.predict(X_train), y_train))
                    DATA_dict[estimator]["mae_test"].append(mean_absolute_error(optEstimator.predict(X_test), y_test))
                    DATA_dict[estimator]["R2"].append(r2_score(y_test, optEstimator.predict(X_test)))
                    DATA_dict[estimator]["Pearson"].append(pearsonr(y_test, optEstimator.predict(X_test)))
                    DATA_dict[estimator]["Spearman"].append(spearmanr(y_test, optEstimator.predict(X_test)))
                    
                    
            model_line=0
            for estimator in name_models:
                
                models_scan.loc[(index_line + model_line)]=[(estimator+"_"+str(NNB_radius)+"_"+str(ring_radius)),
                                                                int(sel),
                                                                int(NNB_radius),
                                                                int(ring_radius),                
                                                                np.mean(DATA_dict[estimator]["mae_train"]),
                                                                np.std(DATA_dict[estimator]["mae_train"]),
                                                                np.mean(DATA_dict[estimator]["mae_test"]),
                                                                np.std(DATA_dict[estimator]["mae_test"]),
                                                                np.mean(DATA_dict[estimator]["EOKfold"]),
                                                                np.std(DATA_dict[estimator]["EOKfold"]),
                                                                np.mean(DATA_dict[estimator]["R2"]),
                                                                np.median(DATA_dict[estimator]["Pearson"]),
                                                                np.median(DATA_dict[estimator]["Spearman"]),
                                                                len(selected_features),
                                                                selected_features,
                                                                list(DATA_dict[estimator]["mae_test"]),
                                                                list(DATA_dict[estimator]["R2"]),
                                                                list(DATA_dict[estimator]["EOKfold"])
                                                                ]
                model_line+=1
            

            models_scan.to_excel(path_dir1+"AntonioM/Models_scan"+file_input_name+selector+".xlsx")  
        
            index_line+=6 
            
