# Em Predict.py is the python script for the ML pipeline used to test the performance of the various ML models

import pandas as pd
import numpy as np
import os
import logging
import warnings
import time
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from utils import RHCF, RemoveOutliar
from ML_models import models_dict

warnings.filterwarnings("ignore")

# initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %% define models
name_models = ["LR", "GPR", "KNR", "SVR", "RF", "XGB"]  # list of models to test
models_with_scaling = [
    "LR",
    "GPR",
    "KNR",
    "SVR",
]  # list of models with features scaling (StandardScaler)
models_with_fs = ["LR", "GPR", "KNR", "SVR"]  # list of models with features selection
features_list = []  # eventually subset of features to use

n_jobs = 4  # number of processes in parallel
path_inputs = "dataset_features/"
name_output_proteins = "analysis_dataset"
name_output_result = "results_dataset"
path_dir_output = "outputs/"

# range of radii combination dataset to test
list_Bar_radius = np.arange(8, 17)
list_Ring_radius = np.arange(3, 7)

# ML parameters
n_repeat = 10  # out nested cross validation
split_tuning = 5  # inner nested corss validation used for hyperparameters tuning
CV = 5  # cross validation for hyperparameters tuning
opt = "neg_mean_absolute_error"  # evaluation metric for hyperparameters tuning
covariation = 0.99  # covariance threshold
alpha = 10  # alpha parameter for features selection
l1_ratio = 0.75  # hyperparameter for ElasticNet method (if l1_ratio = 0 the penalty is an L2 penalty; if l1_ratio = 1 it is an L1 penalty)
max_iter = 1000  # hyperparameter for ElasticNet method

data = pd.read_excel("data/dataset.xlsx", index_col=0)
proteins = list(data["PDB"])  # list of PDB ID to be considered

# inizialize pd.dataframe which will be used to save results row by row
models_scan = pd.DataFrame(
    columns=[
        "name_model_radius",
        "estimator",
        "bar_radius",
        "ring_radius",
        "MAE_train",
        "sd_train",
        "MAE_test",
        "sd_test",
        "mean_error_kfold",
        "sd_error_kfold",
        "RMSE",
        "sd_RMSE",
        "R2",
        "sd_R2",
        "Pearson",
        "sd_Pearson",
        "Spearman",
        "sd_Spearman",
        "Mae_test_list",
        "Error_Kfold_list",
        "RMSE_list",
        "R2_test_list",
        "Pearson_test_list",
        "Spearman_test_list",
        "Best_params",
        "selected_features",
        "n_features",
    ]
)

dict_proteins = dict()

en = ElasticNet(
    max_iter=max_iter, alpha=alpha, l1_ratio=l1_ratio
)  # inizialize features selector
estimator_feature = SelectFromModel(en)
imp = SimpleImputer(
    missing_values=np.nan, strategy="mean"
)  # method used to fill missing value
df_proteins = None
index_line = 0  # index rows for model scan dataframe

for bar_radius in list_Bar_radius:
    for ring_radius in list_Ring_radius:
        dict_proteins = dict()
        # %%
        file_name = f"dataset_protein_{str(bar_radius)}_{str(ring_radius)}.xlsx"
        logger.info(f"processing: {file_name}")

        # Upload dataset
        df_pmOrig = pd.read_excel(
            os.path.join(path_inputs, file_name), sheet_name="Sheet1", index_col=0
        ).set_index("PDB ID")
        df_pm = df_pmOrig.copy()
        for estimator in name_models:
            dict_proteins[estimator] = dict()
            cont = 0
            for key in df_pm.index:
                dict_proteins[estimator][f"{key}_{str(cont)}"] = 0
                cont += 1

        if len(features_list) != 0:
            df_pm = df_pm.loc[:, features_list]

        # Input/output
        X = df_pm.iloc[:, 1:].values  # all values except Em
        y = df_pm.iloc[:, 0].values  # Em values
        labels = list(dict_proteins[name_models[0]].keys())
        labels = np.array(labels)

        # Hyperparameter Tuning
        logger.info("Hyperparameter Tuning")

        if df_proteins is None:
            df_proteins = pd.DataFrame(index=labels)

        DATA_dict = {
            name_model: {
                "mae_train": [],
                "mae_test": [],
                "RMSE": [],
                "R2": [],
                "Pearson": [],
                "Spearman": [],
                "EOKfold": [],
            }
            for name_model in name_models
        }

        startTime = time.time()

        i = 0
        # nested cross validation outer loop
        for j in range(n_repeat):
            kfold = KFold(n_splits=split_tuning, random_state=j, shuffle=True)
            # nested cross validation inner loop
            for train_index, test_index in kfold.split(X):
                logger.info(f"Split_train_test: {i}")

                i = i + 1
                # get index for train and test set
                X_train = X[train_index, :]
                y_train = y[train_index]
                labels_train = labels[train_index]
                X_test = X[test_index, :]
                y_test = y[test_index]
                labels_test = labels[test_index]

                ############## preprocessing operations ##############

                # fill missing value for pH
                fillph = imp.fit(X_train)
                X_train = fillph.transform(X_train)
                X_test = fillph.transform(X_test)
                # remove highrelated features
                remove_hcf = RHCF(covariation=covariation).fit(X_train)
                X_train = remove_hcf.transform(X_train)
                X_test = remove_hcf.transform(X_test)
                X_train2 = X_train.copy()
                X_test2 = X_test.copy()

                for estimator in name_models:
                    # features scaling if needed
                    if estimator in models_with_scaling:
                        sc = StandardScaler().fit(X_train2)
                        X_train2 = sc.transform(X_train2)
                        X_test2 = sc.transform(X_test2)

                    # inizialize pipeline as list
                    pipeline = []
                    # feature selection
                    if estimator in models_with_fs:
                        pipeline.append(("feature_selection", estimator_feature))

                    # add specific model to pipeline
                    pipeline.append(("regressor", models_dict[estimator][0]))

                    # pipe object with information from pipiline list
                    pipe = Pipeline(pipeline)

                    # useful for applying a non-linear transformation to the target y
                    treg = TransformedTargetRegressor(regressor=pipe, transformer=None)

                    optEstimator = GridSearchCV(
                        treg,
                        models_dict[estimator][1],
                        scoring=opt,
                        cv=CV,
                        n_jobs=n_jobs,
                    )

                    _ = optEstimator.fit(X_train2, y_train)
                    # save list of different statistic evaluation metric for each model tested
                    DATA_dict[estimator]["EOKfold"].append(
                        np.abs(optEstimator.best_score_)
                    )
                    DATA_dict[estimator]["mae_train"].append(
                        mean_absolute_error(optEstimator.predict(X_train2), y_train)
                    )
                    DATA_dict[estimator]["mae_test"].append(
                        mean_absolute_error(optEstimator.predict(X_test2), y_test)
                    )
                    DATA_dict[estimator]["RMSE"].append(
                        np.sqrt(
                            mean_squared_error(optEstimator.predict(X_test2), y_test)
                        )
                    )
                    DATA_dict[estimator]["R2"].append(
                        r2_score(y_test, optEstimator.predict(X_test2))
                    )
                    DATA_dict[estimator]["Pearson"].append(
                        pearsonr(y_test, optEstimator.predict(X_test2))[0]
                    )
                    DATA_dict[estimator]["Spearman"].append(
                        spearmanr(y_test, optEstimator.predict(X_test2))[0]
                    )

                    values = list(np.abs(optEstimator.predict(X_test2) - y_test))
                    for label, value in zip(labels_test, values):
                        dict_proteins[estimator][label] += value

        for label in labels:
            for estimator in name_models:
                dict_proteins[estimator][label] = (
                    dict_proteins[estimator][label] / n_repeat
                )

        for estimator in name_models:
            df_proteins[f"{estimator}_{str(bar_radius)}_{str(ring_radius)}"] = [
                dict_proteins[estimator][key] for key in dict_proteins[estimator].keys()
            ]

        if "Em" not in df_proteins.columns:
            df_proteins.insert(loc=1, column="Em", value=df_pm["Em"].values)

        df_proteins.to_csv(
            path_dir_output + name_output_proteins + ".csv"
        )  # save the file with the model error for each entries of the specific dataset
        model_line = 0

        # model tested out of nested cross validation
        for estimator in name_models:
            X2 = X.copy()
            # pH
            fillph = imp.fit(X2)
            X2 = fillph.transform(X2)
            # remove correlated feature
            remove_hcf = RHCF(covariation=covariation).fit(X2)
            X2 = remove_hcf.transform(X2)
            # scaling
            if estimator in models_with_scaling:
                X2 = StandardScaler().fit(X2).transform(X2)

            pipeline = []
            # feature selection
            if estimator in models_with_fs:
                pipeline.append(("feature_selection", estimator_feature))

            pipeline.append(("regressor", models_dict[estimator][0]))
            pipe = Pipeline(pipeline)

            # useful for applying a non-linear transformation to the target y
            treg = TransformedTargetRegressor(regressor=pipe, transformer=None)

            optEstimator = GridSearchCV(
                treg, models_dict[estimator][1], scoring=opt, cv=CV, n_jobs=n_jobs
            )

            best_model = optEstimator.fit(X2, y)
            best_params = best_model.best_params_

            # feature selection
            if estimator in models_with_fs:
                alphaTot = best_model.best_params_[
                    "regressor__feature_selection__estimator__alpha"
                ]
                l1_ratioTot = best_model.best_params_[
                    "regressor__feature_selection__estimator__l1_ratio"
                ]
                enTot = ElasticNet(
                    max_iter=max_iter, alpha=alphaTot, l1_ratio=l1_ratioTot
                )
                estimator_featureTot = SelectFromModel(enTot)
                fs = estimator_featureTot.fit(X2, y)

                C = np.array(np.abs(fs.estimator_.coef_) > 0)
                selected_features = []

                features_index = [1 + el for el in remove_hcf.to_keep]
                for el, el2 in zip(df_pm.columns[features_index], C):
                    if el2 == True:
                        selected_features.append(el)

            else:
                selected_features = []

            models_scan.loc[(index_line + model_line)] = [
                (f"estimator+_{str(bar_radius)}_{str(ring_radius)}"),
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
                len(selected_features),
            ]
            model_line += 1

        executionTime = time.time() - startTime
        print("Execution time in seconds: " + str(executionTime))

        models_scan.to_excel(
            os.path.join(path_dir_output, f"{name_output_result}.xlsx")
        )

        index_line += len(name_models)
