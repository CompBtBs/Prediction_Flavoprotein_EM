import numpy as np
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# dict for models with hyperparameters grid which will be tuned with GridSearch optimization

models_dict = {
    "LR": [
        LinearRegression(),
        {
            "regressor__feature_selection__estimator__alpha": [10, 100],
            "regressor__feature_selection__estimator__l1_ratio": [0.5, 0.75, 1],
        },
    ],
    "KNR": [
        KNeighborsRegressor(),
        {
            "regressor__feature_selection__estimator__alpha": [10, 100],
            "regressor__feature_selection__estimator__l1_ratio": [0.5, 0.75, 1],
            "regressor__regressor__n_neighbors": [2, 3, 4, 5, 6, 7],
            "regressor__regressor__metric": ["euclidean", "manhattan"],
            "regressor__regressor__weights": ["uniform", "distance"],
        },
    ],
    "SVR": [
        SVR(kernel="rbf"),
        {
            "regressor__feature_selection__estimator__alpha": [10, 100],
            "regressor__feature_selection__estimator__l1_ratio": [0.5, 0.75, 1],
            "regressor__regressor__C": np.logspace(-3, 3, 7),
            "regressor__regressor__gamma": np.logspace(-3, 3, 7),
        },
    ],
    "GPR": [
        GaussianProcessRegressor(
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(1.0, length_scale_bounds="fixed")
        ),
        {
            "regressor__feature_selection__estimator__alpha": [10, 100],
            "regressor__feature_selection__estimator__l1_ratio": [0.5, 0.75, 1],
            "regressor__regressor__alpha": np.logspace(-2, 2, 5),
            "regressor__regressor__kernel__k1__constant_value": np.logspace(-2, 2, 5),
            "regressor__regressor__kernel__k2__length_scale": np.logspace(-2, 2, 5),
        },
    ],
    "RF": [
        RandomForestRegressor(),
        {
            "regressor__regressor__n_estimators": [100, 150, 200],
            "regressor__regressor__max_features": ["auto", "sqrt", "log2"],
            "regressor__regressor__max_depth": [3, 4, 5],
        },
    ],
    "XGB": [
        XGBRegressor(),
        {
            "regressor__regressor__learning_rate": [0.01, 0.1, 0.2, 0.4],
            "regressor__regressor__max_depth": [3, 4, 5],
            "regressor__regressor__min_child_weight": [1, 5, 10],
            "regressor__regressor__n_estimators": [100, 150, 200],
        },
    ],
}
