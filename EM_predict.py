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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV,LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
import time

# In[2]:

import warnings
warnings.filterwarnings('ignore')

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
                                  "Best_params","selected_features","n_features"])

covariation=0.99
alpha=100                    #alpha paramter for feature selection
list_NNB_radius=np.arange(8,17)     
list_N5_radius=np.arange(3,6)
CV=5 #cross validation for hyperparameters tuning
opt="neg_mean_absolute_error" #evaluation metric for hyperparameters tuning
split_tuning=50 #number of split for hyperparameters tuning
test_size=0.2 #train test split 
n_jobs=4       #number of processes in parallel
file_input_name ="proteins" #or proteins 
models_dict={"LR":[LinearRegression(),{}],
             'GPR':[GaussianProcessRegressor(ConstantKernel(1.0,constant_value_bounds="fixed") * RBF(1.0,length_scale_bounds="fixed")),                    
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
                    #"regressor__learning_rate" : [0.01,0.1,0.2,0.4],
                    "regressor__max_depth" : [3,4,5],
                    #"regressor__min_child_weight" : [1,5,10],
                    "regressor__n_estimators" : [100,150,200]
                    }]
             } 


dict_proteins=dict()

#add dict for selector in gridsearch
for estimator in models_dict.keys():
        #models_dict[estimator][1]["selector__estimator__alpha"]=[10]    
        models_dict[estimator][1]["selector__estimator__l1_ratio"]=[0.75,1]

name_models=["KNR"]




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
            for key in df_pm.index:
                dict_proteins[key]=0
                dict_proteins_cont[key]=0
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
            # Hyperparameter Tuning
            print( "hyperparameter tuning")
            # In[]:

            DATA_dict={name_model:{"mae_train":[],"mae_test":[],"mae_lists":[],"RMSE":[],"R2":[],"Pearson":[],"Spearman":[],"EOKfold":[]} for name_model in name_models}

            startTime = time.time ()
            for i in range(split_tuning):
                print("Split_train_test:",i)
                indices = np.arange(X.shape[0])
                labels = list(dict_proteins.keys())
                X_train, X_test, y_train, y_test,labels_train,labels_test,indices_train,indices_test= train_test_split(X, y, labels, indices,test_size=test_size,random_state=i)

                estimator_feature=ElasticNet(max_iter=1000,alpha=alpha)


                for estimator in name_models:

                    pipe = Pipeline([
                                     ('scaler', StandardScaler()),
                                     ('selector', SelectFromModel(estimator=estimator_feature)),
                                     ('regressor', models_dict[estimator][0])
                                    ])
                    
                    optEstimator = GridSearchCV(pipe, models_dict[estimator][1],
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
                        dict_proteins[label]+=value
                        dict_proteins_cont[label]+=1
                        
                    for label in labels_test:
                        dict_proteins[label]=dict_proteins[label]/dict_proteins_cont[label]
                    DATA_dict[estimator]["mae_lists"]=dict_proteins
                    DATA_dict[estimator]["mae_conts"]=dict_proteins_cont
                    
                    
                    DATA_dict[estimator]["RMSE"].append(mean_squared_error(optEstimator.predict(X_test), y_test))
                    DATA_dict[estimator]["R2"].append(r2_score(y_test, optEstimator.predict(X_test)))
                    DATA_dict[estimator]["Pearson"].append(pearsonr(y_test, optEstimator.predict(X_test))[0])
                    DATA_dict[estimator]["Spearman"].append(spearmanr(y_test, optEstimator.predict(X_test))[0])
                    
            model_line=0
            for estimator in name_models:

                pipe = Pipeline([
                                  ('scaler', StandardScaler()),
                                  ('selector', SelectFromModel(estimator=estimator_feature)),
                                  ('regressor', models_dict[estimator][0])
                                ])


                optEstimator = GridSearchCV(pipe, models_dict[estimator][1],
                                            scoring=opt,cv=CV,n_jobs=n_jobs
                                            )
                
                
                best_model=optEstimator.fit(X,y)
                best_params=best_model.best_params_
                l1_ratio=best_model.best_params_["selector__estimator__l1_ratio"]

                X_scaling=StandardScaler().fit(X).transform(X)     
                selection=ElasticNet(max_iter=1000,l1_ratio=l1_ratio,alpha=alpha).fit(X_scaling,y)     
                C=np.array(np.abs(selection.coef_)>0)
                
                selected_features=[]
                for el,el2 in zip(df_pm.columns[1:],C):
                    if el2==True:
                        selected_features.append(el) 
                
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
                                                                list(selected_features),
                                                                len(selected_features)
                                                                ]
                model_line+=1
                
                a=models_scan["MAE_lists"].iloc[0]
                b=models_scan["MAE_conts"].iloc[0]
                a1=[a[key] for key in a.keys()]
                b1=[b[key] for key in b.keys()]
                plt.figure()
                plt.scatter(a1,b1)
            
            executionTime = (time.time () - startTime) 
            print ('Execution time in seconds: ' + str (executionTime)) 
            
            models_scan.to_excel(path_dir_output+file_input_name+".xlsx")  
        
            index_line+=6 
            
                        

