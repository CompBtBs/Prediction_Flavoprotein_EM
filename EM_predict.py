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
import sys
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from scipy.stats import pearsonr,spearmanr
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor,LocalOutlierFactor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV,LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
import time
from sklearn.ensemble import IsolationForest
from sklearn.compose import TransformedTargetRegressor
# In[2]:

import warnings
#warnings.filterwarnings('ignore')

# In[5]:

path_inputs="dataset_features/"
path_dir_output="outputs/"
# In[ciclo_per_più_dataset]:

models_scan=pd.DataFrame(columns=["name_model_radius",
                                  "estimator","NNB_radius","N5_radius",
                                  "MAE_train","sd_train",
                                  "MAE_test","sd_test","MAE_lists","MAE_conts",
                                  "mean_error_kfold","sd_error_kfold","RMSE","sd_RMSE","R2","Pearson","Spearman",
                                  "Mae_test_list","Error_Kfold_list","RMSE_list",
                                  "R2_test_list","Pearson_test_list","Spearman_test_list",
                                  "Best_params",
                                  #"selected_features","n_features"
                                  ])

alpha=10                    #alpha paramter for feature selection
list_NNB_radius=np.arange(8,17)     
list_N5_radius=np.arange(3,6)
CV=5 #cross validation for hyperparameters tuning
opt="neg_mean_absolute_error" #evaluation metric for hyperparameters tuning
split_tuning=5                #number of split for hyperparameters tuning (quindi 80 e 20)
n_repeat=10                    #ripetizioni dell'esperimento
#test_size=0.2                 #train test split 
n_jobs=4       #number of processes in parallel
file_input_name ="proteins" #or proteins 
models_dict={"LR":[LinearRegression(),{}],
             'GPR':[GaussianProcessRegressor(ConstantKernel(1.0,constant_value_bounds="fixed") * RBF(1.0,length_scale_bounds="fixed")),                    
                    {
                    'regressor__regressor__alpha':np.logspace(-2, 2, 5),
                    'regressor__regressor__kernel__k1__constant_value': np.logspace(-2, 2, 5),
                    'regressor__regressor__kernel__k2__length_scale': np.logspace(-2, 2, 5)
                    }],
             'SVR':[SVR(kernel="rbf"),                   
                    {
                    'regressor__regressor__C': np.logspace(-3,3,7),
                    'regressor__regressor__gamma': np.logspace(-3,3,7)
                    #'regressor__epsilon': np.logspace(-1,0,2)
                    }],
             'KNR':[KNeighborsRegressor(),
                    {           
                    'regressor__regressor__n_neighbors': [2, 3, 4, 5, 6, 7],
                    'regressor__regressor__metric': ['euclidean','manhattan'],
                    'regressor__regressor__weights':  ["uniform", "distance"]
                    }],
             "RF":[RandomForestRegressor(),
                   {
                    'regressor__regressor__n_estimators': [100,150,200],
                    'regressor__regressor__max_features': ['auto', 'sqrt','log2'],
                    'regressor__regressor__max_depth': [3, 4, 5]
                    }],
             "XGB":[XGBRegressor(),
                    {
                    "regressor__regressor__learning_rate" : [0.01,0.1,0.2,0.4],
                    "regressor__regressor__max_depth" : [3,4,5],
                    "regressor__regressor__min_child_weight" : [1,5,10],
                    "regressor__regressor__n_estimators" : [100,150,200]
                    }]
             } 


dict_proteins=dict()

#add dict for selector in gridsearch
for estimator in ["LR","KNR","SVR","GPR"]:
    #models_dict[estimator][1]["regressor__selector__estimator__alpha"]=[10,100]    
    models_dict[estimator][1]["regressor__selector__estimator__l1_ratio"]=[0.5,0.75,1]

name_models=["LR","KNR","GPR","SVR","RF","XGB"]

df_protein_organism=pd.read_csv("organism.csv",index_col=0)

###############################################################################################
# In[]:

index_line=0 #index riga for model scan dataset 

for NNB_radius in list_NNB_radius: 
    for N5_radius in list_N5_radius:
            dict_proteins=dict()
            dict_proteins_cont=dict()
#%%
            file_name="dataset_"+file_input_name+"_"+str(NNB_radius)+"_"+str(N5_radius)+".xlsx"
            print("dataset_"+file_input_name+"_"+str(NNB_radius)+"_"+str(N5_radius))
            
            # Upload dataset
            #file_name="database_chains_8_4.xlsx"
            df_pm=pd.read_excel(path_inputs+file_name,sheet_name="Sheet1",index_col=0)
            df_pm=df_pm.drop_duplicates()
            #df_pm=df_pm.loc[[el for el in df_pm.index if el!="1M6I"]]
            for estimator in name_models:
                dict_proteins[estimator]=dict()
                dict_proteins_cont[estimator]=dict()
                for key in df_pm.index:
                    dict_proteins[estimator][key]=0
                    dict_proteins_cont[estimator][key]=0
                    
            df_pm=df_pm.reset_index()
            df_pm=df_pm.iloc[:,1:]
            if "Protein_name" in df_pm.columns:
                df_pm=df_pm.drop(["Protein_name"],axis=1) 
            
            columns_to_keep_=[el for el in df_pm.columns if "%" not in el and "_mean" not in el]
            df_pm=df_pm.loc[:,columns_to_keep_]

            # Restituisce in output il dataset per fare Machine Learning
                        
            # In[]:
            # Input/output

            X=df_pm.iloc[:,1:].values
            y=df_pm.iloc[:,0].values

            #clf = IsolationForest(random_state=0).fit(X)
            print(len(y))
            #y=y[clf.predict(X)==1]
            #print(len(y))
            labels = list(dict_proteins[name_models[0]].keys())
            labels=np.array(labels)
            #labels=labels[clf.predict(X)==1]
            #X=X[clf.predict(X)==1,:] 
            # In[]:
            # Hyperparameter Tuning
            print( "hyperparameter tuning")
            # In[]:
            df_proteins=pd.DataFrame(index=labels)


            DATA_dict={name_model:{"mae_train":[],"mae_test":[],"mae_lists":[],"RMSE":[],"R2":[],"Pearson":[],"Spearman":[],"EOKfold":[]} for name_model in name_models}

            startTime = time.time ()
            #for i in range(split_tuning):
            #    X_train, X_test, y_train, y_test,labels_train,labels_test= train_test_split(X, y, labels,test_size=test_size,random_state=i)
            i=0
            for j in range(n_repeat):
                #ripeto la neste n volte ma così facendo tutte le proteine vengono visitate lo stesso numero di volte
                kfold=KFold(n_splits=split_tuning, random_state=j, shuffle=True)
                #eseguo una nested 
                
                for train_index, test_index  in kfold.split(X):
                    print("Split_train_test:",i)
                    i=i+1
                    X_train=X[train_index,:]
                    y_train=y[train_index ]
                    labels_train=labels[train_index]
                    X_test=X[test_index,:]
                    y_test=y[test_index]                    
                    labels_test=labels[test_index]                         
                        
                    
                    estimator_feature=ElasticNet(max_iter=1000,alpha=alpha)
    
                    
                    for estimator in name_models:
                        pipeline=[]
                        if estimator in ["LR","SVR","GPR","KNR"]:
                            #lo scaling si fai solo per LR,SVR e GPR
                            pipeline.append(('scaler', StandardScaler()))
                        if  estimator in ["LR","SVR","KNR","GPR"]: 
                            #selziono le feature solo per LR,GPR e SVR
                            pipeline.append(('selector', SelectFromModel(estimator=estimator_feature)))
                            
                        pipeline.append(('regressor', models_dict[estimator][0]))
                        pipe=Pipeline(pipeline)
                        
                        treg=TransformedTargetRegressor(regressor=pipe,transformer=None) #trasforma eventualmente la variabile y
                        
                        optEstimator = GridSearchCV(treg, models_dict[estimator][1],
                                                    scoring=opt,cv=CV,
                                                    n_jobs=n_jobs
                                                    )
                                                    
                        _= optEstimator.fit(X_train, y_train)
                        #print(optEstimator.best_estimator_.get_params())
                        DATA_dict[estimator]["EOKfold"].append(np.abs(optEstimator.best_score_))
                        DATA_dict[estimator]["mae_train"].append(mean_absolute_error(optEstimator.predict(X_train), y_train))
                        DATA_dict[estimator]["mae_test"].append(mean_absolute_error(optEstimator.predict(X_test), y_test))
                        
                        values=list(np.abs(optEstimator.predict(X_test)-y_test))
    
                        for label,value in zip(labels_test,values):
                            dict_proteins[estimator][label]+=value
                            dict_proteins_cont[estimator][label]+=1
                            
                        for label in labels_test:
                            dict_proteins[estimator][label]=dict_proteins[estimator][label]/dict_proteins_cont[estimator][label]
                        
                        DATA_dict[estimator]["mae_lists"]=dict_proteins[estimator]
                        DATA_dict[estimator]["mae_conts"]=dict_proteins_cont[estimator]
                        
                        
                        DATA_dict[estimator]["RMSE"].append(np.sqrt(mean_squared_error(optEstimator.predict(X_test), y_test)))
                        DATA_dict[estimator]["R2"].append(r2_score(y_test, optEstimator.predict(X_test)))
                        DATA_dict[estimator]["Pearson"].append(pearsonr(y_test, optEstimator.predict(X_test))[0])
                        DATA_dict[estimator]["Spearman"].append(spearmanr(y_test, optEstimator.predict(X_test))[0])
                        
                        df_proteins[estimator]=[dict_proteins[estimator][key] for key in dict_proteins[estimator].keys()]
                df_proteins["organism"]=df_protein_organism["organism"]
                #df_proteins=df_proteins.loc[[el for el in df_proteins.index if el!="1M6I"]]
                model_line=0
                for estimator in name_models:
    
                    pipe = Pipeline([
                                      ('scaler', StandardScaler()),
                                      ('selector', SelectFromModel(estimator=estimator_feature)),
                                      ('regressor', models_dict[estimator][0])
                                    ])
    
                    treg=TransformedTargetRegressor(regressor=pipe,transformer=StandardScaler())
                    optEstimator = GridSearchCV(treg, models_dict[estimator][1],
                                                scoring=opt,cv=CV,n_jobs=n_jobs
                                                )
                    
                    
                    best_model=optEstimator.fit(X,y)
                    best_params=best_model.best_params_
                    #l1_ratio=best_model.best_params_["regressor__selector__estimator__l1_ratio"]
    
                    X_scaling=StandardScaler().fit(X).transform(X)     
                    #selection=ElasticNet(max_iter=1000,l1_ratio=l1_ratio,alpha=alpha).fit(X_scaling,y)     
                    #C=np.array(np.abs(selection.coef_)>0)
                    
                    #selected_features=[]
                    #for el,el2 in zip(df_pm.columns[1:],C):
                    #    if el2==True:
                    #        selected_features.append(el) 
                    
                    models_scan.loc[(index_line + model_line)]=[(estimator+"_"+str(NNB_radius)+"_"+str(N5_radius)),
                                                                    estimator,
                                                                    int(NNB_radius),
                                                                    int(N5_radius),                
                                                                    np.mean(DATA_dict[estimator]["mae_train"]),
                                                                    np.std(DATA_dict[estimator]["mae_train"]),
                                                                    np.mean(DATA_dict[estimator]["mae_test"]),
                                                                    np.std(DATA_dict[estimator]["mae_test"]),
                                                                    DATA_dict[estimator]["mae_lists"],
                                                                    DATA_dict[estimator]["mae_conts"],
                                                                    np.mean(DATA_dict[estimator]["EOKfold"]),
                                                                    np.std(DATA_dict[estimator]["EOKfold"]),
                                                                    np.mean(DATA_dict[estimator]["RMSE"]),
                                                                    np.std(DATA_dict[estimator]["RMSE"]),
                                                                    np.mean(DATA_dict[estimator]["R2"]),
                                                                    np.mean(DATA_dict[estimator]["Pearson"]),
                                                                    np.mean(DATA_dict[estimator]["Spearman"]),
                                                                    list(DATA_dict[estimator]["mae_test"]),
                                                                    list(DATA_dict[estimator]["EOKfold"]),
                                                                    list(DATA_dict[estimator]["RMSE"]),
                                                                    list(DATA_dict[estimator]["R2"]),
                                                                    list(DATA_dict[estimator]["Pearson"]),
                                                                    list(DATA_dict[estimator]["Spearman"]),   
                                                                    best_params,
                                                                    #list(selected_features),
                                                                    #len(selected_features)
                                                                    ]
                    model_line+=1
                    

            
            executionTime = (time.time () - startTime) 
            print ('Execution time in seconds: ' + str (executionTime)) 
            
            models_scan.to_excel(path_dir_output+file_input_name+".xlsx")  
            df_proteins.to_csv(path_dir_output+"analysis_proteins_NNB_"+str(NNB_radius)+"_N5_"+str(N5_radius)+".csv")
            index_line+=len(name_models)
            
                        

