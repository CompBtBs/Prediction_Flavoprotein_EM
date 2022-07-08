# -*- coding: utf-8 -*-



# In[]:
from scipy.stats import pearsonr,spearmanr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
from sklearn.ensemble import IsolationForest
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import warnings
import time
from utils import RHCF,RemoveOutliar

# In[2]:

warnings.filterwarnings('ignore')

# In[5]: models
name_models=["LR","GPR","KNR","SVR","RF","XGB"]
models_with_scaling=["LR","GPR","KNR","SVR"]    
models_with_fs=["LR","GPR","KNR","SVR"]         
select_features=[]#feature_selected()
filter_proteins=True
############
n_jobs=4                      #number of processes in parallel
path_inputs="dataset_features/"
name_output_proteins="analysis_dataset"
name_output_result="results_dataset"
path_dir_output="outputs/"
list_Bar_radius=np.arange(8,17)     
list_Ring_radius=np.arange(3,7)

###parameters
n_repeat=10                     #ripetizioni dell'esperimento
split_tuning=5                  #number of split for hyperparameters tuning (quindi 80 e 20)
CV=5                            #cross validation for hyperparameters tuning
opt="neg_mean_absolute_error"   #evaluation metric for hyperparameters tuning
covariation=0.99                #covariance threshold
alpha=10                        #alpha paramter for feature selection
l1_ratio=0.75                   #
max_iter=1000

###############################################################################################
#%%
df_data=pd.read_excel("data/dataset.xlsx",index_col=0)
proteins=list(df_data.index)


models_scan=pd.DataFrame(columns=["name_model_radius",
                                  "estimator","bar_radius","ring_radius",
                                  "MAE_train","sd_train",
                                  "MAE_test","sd_test",
                                  "mean_error_kfold","sd_error_kfold","RMSE","sd_RMSE","R2","sd_R2",
                                  "Pearson","sd_Pearson","Spearman","sd_Spearman",
                                  "Mae_test_list","Error_Kfold_list","RMSE_list",
                                  "R2_test_list","Pearson_test_list","Spearman_test_list",
                                  "Best_params",
                                  "selected_features","n_features"
                                  ])

models_dict={"LR":[LinearRegression(),{
                    'regressor__feature_selection__estimator__alpha':[10,100],
                    'regressor__feature_selection__estimator__l1_ratio':[0.5,0.75,1],
    
    }],
             'KNR':[KNeighborsRegressor(),
                    {           
                    'regressor__feature_selection__estimator__alpha':[10,100],
                    'regressor__feature_selection__estimator__l1_ratio':[0.5,0.75,1],
                    'regressor__regressor__n_neighbors': [2, 3, 4, 5, 6, 7],
                    'regressor__regressor__metric': ['euclidean','manhattan'],
                    'regressor__regressor__weights':  ["uniform", "distance"]
                    }],
    
             'SVR':[SVR(kernel="rbf"),                   
                    {
                    'regressor__feature_selection__estimator__alpha':[10,100],
                    'regressor__feature_selection__estimator__l1_ratio':[0.5,0.75,1],
                    'regressor__regressor__C': np.logspace(-3,3,7),
                    'regressor__regressor__gamma': np.logspace(-3,3,7)
                    }],   
             'GPR':[GaussianProcessRegressor(ConstantKernel(1.0,constant_value_bounds="fixed") * RBF(1.0,length_scale_bounds="fixed")),                    
                    {
                    'regressor__feature_selection__estimator__alpha':[10,100],
                    'regressor__feature_selection__estimator__l1_ratio':[0.5,0.75,1],                    
                    'regressor__regressor__alpha':np.logspace(-2, 2, 5),
                    'regressor__regressor__kernel__k1__constant_value': np.logspace(-2, 2, 5),
                    'regressor__regressor__kernel__k2__length_scale': np.logspace(-2, 2, 5)
                    }],

             "RF":[RandomForestRegressor(),
                   {
                    'regressor__regressor__n_estimators': [100,150,200],
                    'regressor__regressor__max_features': ['auto', 'sqrt','log2'],
                    'regressor__regressor__max_depth': [3,4, 5]
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
# In[]:
en=ElasticNet(max_iter=max_iter,alpha=alpha,l1_ratio=l1_ratio)
estimator_feature=SelectFromModel(en)
imp = SimpleImputer(missing_values=np.nan,strategy="mean")
df_proteins=None
index_line=0 #index rows

for bar_radius in list_Bar_radius: 
    for ring_radius in list_Ring_radius:
            dict_proteins=dict()
#%%
            file_name="dataset_protein_"+str(bar_radius)+"_"+str(ring_radius)+".xlsx"
            print(file_name)
            
            # Upload dataset
            df_pmOrig=pd.read_excel(path_inputs+file_name,sheet_name="Sheet1",index_col=0).set_index("PDB ID") 
            df_pm=df_pmOrig.copy()
            for estimator in name_models:
                dict_proteins[estimator]=dict()
                cont=0
                for key in df_pm.index:
                    dict_proteins[estimator][key+"_"+str(cont)]=0
                    cont+=1
                    
            if len(select_features)!=0:
                df_pm=df_pm.loc[:,select_features]

            # In[]:
            # Input/output
            X=df_pm.iloc[:,1:].values
            y=df_pm.iloc[:,0].values
            labels = list(dict_proteins[name_models[0]].keys())
            labels=np.array(labels)
            # In[]:
            # Hyperparameter Tuning
            print( "hyperparameter tuning")
            # In[]:
            if df_proteins is None:
                df_proteins=pd.DataFrame(index=labels)


            DATA_dict={name_model:{"mae_train":[],"mae_test":[],"RMSE":[],"R2":[],"Pearson":[],"Spearman":[],"EOKfold":[]} for name_model in name_models}

            startTime = time.time ()
            
            i=0
            
            for j in range(n_repeat):               
                kfold=KFold(n_splits=split_tuning, random_state=j, shuffle=True)
                
                for train_index, test_index  in kfold.split(X):
                    print("Split_train_test:",i)
                    i=i+1
                    #prelevo gli indici di train e test
                    X_train=X[train_index,:]
                    y_train=y[train_index ]
                    labels_train=labels[train_index]
                    X_test=X[test_index,:]
                    y_test=y[test_index]                    
                    labels_test=labels[test_index]                         
                    ########faccio alcune operazioni di pre-preprocessing
                    #faccio il fill di ph
                    fillph=imp.fit(X_train)
                    X_train=fillph.transform(X_train)
                    X_test=fillph.transform(X_test)
                    #tolgo le feature correlate
                    remove_hcf=RHCF(covariation=covariation).fit(X_train)
                    X_train=remove_hcf.transform(X_train)
                    X_test=remove_hcf.transform(X_test)                    
                    X_train2=X_train.copy()
                    X_test2=X_test.copy()                                
                    
                    for estimator in name_models:
                        #eseguo lo scaling delle variabili
                        if estimator in models_with_scaling:
                            sc=StandardScaler().fit(X_train2)
                            X_train2=sc.transform(X_train2)
                            X_test2=sc.transform(X_test2)
    
                        pipeline=[]    
                        #feature selection
                        if  estimator in models_with_fs:
                            pipeline.append(('feature_selection', estimator_feature))
                        
                        #uso un modello specifico
                        pipeline.append(('regressor', models_dict[estimator][0]))
                        pipe=Pipeline(pipeline)
                        
                        #trasforma eventualmente la variabile y
                        treg=TransformedTargetRegressor(regressor=pipe,transformer=None) 
                        
                        optEstimator = GridSearchCV(treg, models_dict[estimator][1],
                                                    scoring=opt,cv=CV,
                                                    n_jobs=n_jobs
                                                    )
                                                    
                        _= optEstimator.fit(X_train2, y_train)
                        DATA_dict[estimator]["EOKfold"].append(np.abs(optEstimator.best_score_))
                        DATA_dict[estimator]["mae_train"].append(mean_absolute_error(optEstimator.predict(X_train2), y_train))
                        DATA_dict[estimator]["mae_test"].append(mean_absolute_error(optEstimator.predict(X_test2), y_test))
                        DATA_dict[estimator]["RMSE"].append(np.sqrt(mean_squared_error(optEstimator.predict(X_test2), y_test)))
                        DATA_dict[estimator]["R2"].append(r2_score(y_test, optEstimator.predict(X_test2)))
                        DATA_dict[estimator]["Pearson"].append(pearsonr(y_test, optEstimator.predict(X_test2))[0])
                        DATA_dict[estimator]["Spearman"].append(spearmanr(y_test, optEstimator.predict(X_test2))[0])
                        
                        values=list(np.abs(optEstimator.predict(X_test2)-y_test))
                        for label,value in zip(labels_test,values):
                            dict_proteins[estimator][label]+=value
                            
            
            for label in labels:
                for estimator in name_models:
                    dict_proteins[estimator][label]=dict_proteins[estimator][label]/n_repeat
                
            for estimator in name_models:
                df_proteins[estimator+"_"+str(bar_radius)+"_"+str(ring_radius)]=[dict_proteins[estimator][key] for key in dict_proteins[estimator].keys()]
           
            if "Em" not in df_proteins.columns:
                df_proteins.insert(loc =1,
                          column = 'Em',
                          value = df_pm["Em"].values) 
            #df_proteins.to_csv(path_dir_output+name_output_proteins+".csv")
            model_line=0
                
            for estimator in name_models:
                X2=X.copy()
                #ph
                fillph=imp.fit(X2)
                X2=fillph.transform(X2)
                #remove correlated feature
                remove_hcf=RHCF(covariation=covariation).fit(X2)
                X2=remove_hcf.transform(X2)
                #scaling
                if estimator in models_with_scaling:
                    X2=StandardScaler().fit(X2).transform(X2)  
                
                pipeline=[]
                #feature selection
                if  estimator in models_with_fs:
                    pipeline.append(('feature_selection', estimator_feature))                
                
                pipeline.append(('regressor', models_dict[estimator][0]))
                pipe=Pipeline(pipeline)
                
                #trasforma eventualmente la variabile y
                treg=TransformedTargetRegressor(regressor=pipe,transformer=None) 
                
                optEstimator = GridSearchCV(treg, models_dict[estimator][1],
                                            scoring=opt,cv=CV,
                                            n_jobs=n_jobs
                                            )                    
                
                best_model=optEstimator.fit(X2,y)
                best_params=best_model.best_params_

                #feature selection
                if  estimator in models_with_fs:
                    alphaTot=best_model.best_params_['regressor__feature_selection__estimator__alpha']
                    l1_ratioTot=best_model.best_params_['regressor__feature_selection__estimator__l1_ratio']
                    enTot=ElasticNet(max_iter=max_iter,alpha=alphaTot,l1_ratio=l1_ratioTot)
                    estimator_featureTot=SelectFromModel(enTot)
                    fs=estimator_featureTot.fit(X2,y)
                    
                    C=np.array(np.abs(fs.estimator_.coef_)>0)
                    selected_features=[]
                    
                    features_index=[1+el for el in remove_hcf.to_keep]
                    for el,el2 in zip(df_pm.columns[features_index],C):
                        if el2==True:
                            selected_features.append(el) 

                else:
                    selected_features=[]
                    
                models_scan.loc[(index_line + model_line)]=[(estimator+"_"+str(bar_radius)+"_"+str(ring_radius)),
                                                                estimator,
                                                                int(bar_radius),
                                                                int(ring_radius),                
                                                                np.mean(DATA_dict[estimator]["mae_train"]),
                                                                np.std(DATA_dict[estimator]["mae_train"]),
                                                                np.mean(DATA_dict[estimator]["mae_test"]),
                                                                np.std(DATA_dict[estimator]["mae_test"]),
                                                                np.mean(DATA_dict[estimator]["EOKfold"]),
                                                                np.std(DATA_dict[estimator]["EOKfold"]),
                                                                np.mean(DATA_dict[estimator]["RMSE"]),
                                                                np.std(DATA_dict[estimator]["RMSE"]),
                                                                np.mean(DATA_dict[estimator]["R2"]),
                                                                np.std(DATA_dict[estimator]["R2"]),
                                                                np.mean(DATA_dict[estimator]["Pearson"]),
                                                                np.std(DATA_dict[estimator]["Pearson"]),
                                                                np.mean(DATA_dict[estimator]["Spearman"]),
                                                                np.std(DATA_dict[estimator]["Spearman"]),
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
            
            executionTime = (time.time () - startTime) 
            print ('Execution time in seconds: ' + str (executionTime)) 
            
            models_scan.to_excel(path_dir_output+name_output_result+".xlsx")  
            
            index_line+=len(name_models)
            
                        

