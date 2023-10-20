"""
"""
import math
from multiprocessing import cpu_count

import numpy as np
from tensorflow.keras.optimizers import Adam

# import packages for hyperparameters tuning
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Params Start #
n_jobs_cpu = math.floor((cpu_count() / 3))
if n_jobs_cpu < 1:
    n_jobs_cpu = None

# Params: RandomForestClassifier
"""
{
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "class_weight": None,
    "criterion": "gini",
    "max_depth": None,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 100,
    "n_jobs": None,
    "oob_score": False,
    "random_state": None,
    "verbose": 0,
    "warm_start": False,
}
"""
max_iter = [1000, 1500, 2000, 8000]
tol = [0.0001, 0.0005, 0.001]
random_state = [None]
criterion = ["gini", "entropy", "log_loss"]
n_estimators = [100, 150, 200, 300, 500]
max_features = ["sqrt", "log2", None]
max_depth = [None, 1, 3, 6, 9]
max_leaf_nodes = [None, 3, 6, 9]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True]
max_samples = [None, 1, 2, 5]
min_impurity_decrease = [0.0, 0.1]
min_weight_fraction_leaf = [0.0, 0.1]
ccp_alpha = [0.0, 0.01, 0.5]
oob_score = [False, True]
solvers = ["lbfgs", "newton-cg", "liblinear"]
penalty = ["l2", "l1", "elasticnet"]
c_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100]
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
dual = [False, True]

rf_random_grid = {
    "ccp_alpha": ccp_alpha,
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": max_leaf_nodes,
    "bootstrap": bootstrap,
    "class_weight": [None, "balanced", "balanced_subsample"],
    "criterion": criterion,
    "max_samples": max_samples,
    "min_impurity_decrease": min_impurity_decrease,
    "min_weight_fraction_leaf": min_weight_fraction_leaf,
    "n_jobs": [n_jobs_cpu],
    "oob_score": oob_score,
    "random_state": random_state,
    "verbose": [0],
}

rf_gradient_grid = {
    "ccp_alpha": ccp_alpha,
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": max_leaf_nodes,
    "bootstrap": bootstrap,
    "class_weight": [None, "balanced", "balanced_subsample"],
    "criterion": criterion,
    "max_samples": max_samples,
    "min_impurity_decrease": min_impurity_decrease,
    "min_weight_fraction_leaf": min_weight_fraction_leaf,
    "oob_score": oob_score,
    "random_state": random_state,
    "n_jobs": [n_jobs_cpu],
    "verbose": [0],
}

# Params: LogisticRegression
"""
{
    "C": 1.0,
    "class_weight": None,
    "dual": False,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "l1_ratio": None,
    "max_iter": 1500,
    "multi_class": "multinomial",
    "n_jobs": None,
    "penalty": "l2",
    "random_state": None,
    "solver": "lbfgs",
    "tol": 0.0001,
    "verbose": 0,
    "warm_start": False,
}
"""

lr_gradient_grid = {
    "penalty": ["l2", "l1", "elasticnet", None],
    "solver": solvers,
    "C": c_values,
    "class_weight": [None],
    "dual": dual,
    "fit_intercept": [True, False],
    "intercept_scaling": [1],
    "l1_ratio": [None, 0.0, 0.5, 1.0],
    "n_jobs": [n_jobs_cpu],
    "random_state": random_state,
    "tol": tol,
    "verbose": [0],
    "multi_class": ["multinomial", "auto", "ovr"],
    "max_iter": max_iter,
}
lr_random_grid = {
    "penalty": ["l2", "l1", "elasticnet", None],
    "solver": solvers,
    "C": c_values,
    "class_weight": [None],
    "dual": dual,
    "fit_intercept": [True, False],
    "intercept_scaling": [1],
    "l1_ratio": [None, 0.0, 0.5, 1.0],
    "n_jobs": [n_jobs_cpu],
    "random_state": random_state,
    "tol": tol,
    "verbose": [0],
    "multi_class": ["multinomial", "auto", "ovr"],
    "max_iter": max_iter,
}

# Params: LinearSVC
"""
{
    "C": 1.0,
    "class_weight": None,
    "dual": "auto",
    "fit_intercept": True,
    "intercept_scaling": 1,
    "loss": "squared_hinge",
    "max_iter": max_iter,
    "multi_class": "ovr",
    "penalty": "l2",
    "random_state": None,
    "tol": 0.0001,
    "verbose": 0,
}
"""

lsvc_gradient_grid = {
    "C": c_values,
    "max_iter": max_iter,
    "class_weight": [None],
    "dual": dual,
    "fit_intercept": [False, True],
    "intercept_scaling": [1],
    "loss": ["squared_hinge", "hinge"],
    "multi_class": ["ovr", "crammer_singer"],
    "penalty": ["l2", "l1"],
    "random_state": random_state,
    "tol": tol,
    "verbose": [0],
}
lsvc_grid = {
    "C": c_values,
    "max_iter": max_iter,
    "class_weight": [None],
    "dual": dual,
    "fit_intercept": [True, False],
    "intercept_scaling": [1],
    "loss": ["squared_hinge", "hinge"],
    "multi_class": ["ovr", "crammer_singer"],
    "penalty": ["l2", "l1"],
    "random_state": random_state,
    "tol": tol,
    "verbose": [0],
}

# Params: SVC
"""
{
    "C": 1,
    "break_ties": False,
    "cache_size": 200,
    "class_weight": None,
    "coef0": 0.0,
    "decision_function_shape": "ovr",
    "degree": 3,
    "gamma": "scale",
    "kernel": "rbf",
    "max_iter": -1,
    "probability": False,
    "random_state": None,
    "shrinking": True,
    "tol": 0.001,
    "verbose": False,
}
"""

svc_grid = {
    "C": c_values,
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
    "break_ties": [False, True],
    "cache_size": [200, 500],
    "class_weight": [None],
    "coef0": [0.0, 0.1],
    "decision_function_shape": ["ovr", "ovo"],
    "degree": [3],
    "gamma": ["scale", "auto"],
    "max_iter": max_iter,
    "probability": [False, True],
    "random_state": random_state,
    "shrinking": [True, False],
    "tol": tol,
    "verbose": [0],
}
# Params: GaussianNB
gaussian_grid = {
    "var_smoothing": [1e-9, 1e-10, 1e-11, 1e-12],
}

# Params: XGBClassifier
"""
{
    "objective": "binary:logistic",
    "use_label_encoder": None,
    "base_score": None,
    "booster": None,
    "callbacks": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "colsample_bytree": None,
    "early_stopping_rounds": None,
    "enable_categorical": False,
    "eval_metric": None,
    "feature_types": None,
    "gamma": None,
    "gpu_id": None,
    "grow_policy": None,
    "importance_type": None,
    "interaction_constraints": None,
    "learning_rate": None,
    "max_bin": None,
    "max_cat_threshold": None,
    "max_cat_to_onehot": None,
    "max_delta_step": None,
    "max_depth": None,
    "max_leaves": None,
    "min_child_weight": None,
    "missing": nan,
    "monotone_constraints": None,
    "n_estimators": 100,
    "n_jobs": None,
    "num_parallel_tree": None,
    "predictor": None,
    "random_state": None,
    "reg_alpha": None,
    "reg_lambda": None,
    "sampling_method": None,
    "scale_pos_weight": None,
    "subsample": None,
    "tree_method": None,
    "validate_parameters": None,
    "verbosity": None,
}
"""

xgb_gradient_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "learning_rate": [None, 1, 0.5, 0.25, 0.1, 0.05, 0.01],
    "gamma": [None, 0, 0.5, 0.75],
    "reg_alpha": [None, 0, 0.5, 1],
    "reg_lambda": [None, 0.5, 1, 0.75],
    "base_score": [None, 0.2, 0.5, 0.75],
    "eta": [0.2, 0.3, 0.5],
    "colsample_bytree": [None, 0.5, 1, 0.75],
    "booster": [None, "gbtree", "gblinear", "dart"],
    "verbosity": [None, 0],
    "objective": ["binary:logistic", "reg:logistic", "binary:logitraw", "binary:hinge"],
    "n_jobs": [n_jobs_cpu],
}

xgb_random_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "gamma": [None, 0, 0.5, 0.75],
    "learning_rate": [None, 1, 0.5, 0.25, 0.1, 0.05, 0.01],
    "reg_alpha": [None, 0, 0.5, 1],
    "reg_lambda": [None, 0.5, 1, 0.75],
    "colsample_bytree": [None, 0.5, 1, 0.75],
    "min_child_weight": [None, 3, 6, 9, 12],
    "booster": [None, "gbtree", "gblinear", "dart"],
    "base_score": [None, 0.2, 0.5, 0.75],
    "objective": ["binary:logistic", "reg:logistic", "binary:logitraw", "binary:hinge"],
    "n_jobs": [n_jobs_cpu],
}
# Params: GradientBoostingClassifier
"""
{
    "ccp_alpha": 0.0,
    "criterion": "friedman_mse",
    "init": None,
    "learning_rate": 1.0,
    "loss": "log_loss",
    "max_depth": 1,
    "max_features": None,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 100,
    "n_iter_no_change": None,
    "random_state": 0,
    "subsample": 1.0,
    "tol": 0.0001,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": False,
}
"""

gbc_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "max_leaf_nodes": max_leaf_nodes,
    "subsample": [1.0, 0.8],
    "learning_rate": learning_rates,
    "ccp_alpha": ccp_alpha,
    "criterion": ["friedman_mse", "squared_error"],
    "min_impurity_decrease": min_impurity_decrease,
    "min_samples_leaf": min_samples_leaf,
    "min_samples_split": min_samples_split,
    "min_weight_fraction_leaf": min_weight_fraction_leaf,
    "tol": tol,
    "validation_fraction": [0.1],
    "verbose": [0],
}
# Params: AdaBoostClassifier
"""
{
    "algorithm": "SAMME.R",
    "base_estimator": "deprecated",
    "estimator": None,
    "learning_rate": 1.0,
    "n_estimators": 50,
    "random_state": None,
}
"""
ada_grid = {
    "algorithm": ["SAMME.R", "SAMME"],
    "n_estimators": n_estimators,
    "learning_rate": learning_rates,
}

# Params: DecisionTreeClassifier
"""
{
    "ccp_alpha": 0.0,
    "class_weight": None,
    "criterion": "gini",
    "max_depth": None,
    "max_features": None,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "random_state": 0,
    "splitter": "best",
}
"""
dt_grid = {
    "max_features": max_features,
    "max_depth": max_depth,
    "max_leaf_nodes": max_leaf_nodes,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "ccp_alpha": ccp_alpha,
    "class_weight": [None],
    "min_impurity_decrease": min_impurity_decrease,
    "min_weight_fraction_leaf": min_weight_fraction_leaf,
    "splitter": ["best", "random"],
}


# Params: KNeighborsClassifier
"""
{
    "algorithm": "auto",
    "leaf_size": 30,
    "metric": "minkowski",
    "metric_params": None,
    "n_jobs": None,
    "n_neighbors": 3,
    "p": 2,
    "weights": "uniform",
}
"""
n_neighbors = [int(x) for x in np.linspace(start=3, stop=20, num=10)]
leaf_size = [int(x) for x in np.linspace(start=30, stop=80, num=10)]
knn_grid = {
    "n_neighbors": n_neighbors,
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "metric": ["minkowski", "euclidean", "manhattan"],
    "leaf_size": leaf_size,
    "n_jobs": [n_jobs_cpu],
    "p": [2, 1],
}

# Params: MLPClassifier
"""
{
    "activation": "relu",
    "alpha": 0.0001,
    "batch_size": "auto",
    "beta_1": 0.9,
    "beta_2": 0.999,
    "early_stopping": False,
    "epsilon": 1e-08,
    "hidden_layer_sizes": (100,),
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_fun": 15000,
    "max_iter": 100,
    "momentum": 0.9,
    "n_iter_no_change": 10,
    "nesterovs_momentum": True,
    "power_t": 0.5,
    "random_state": 1,
    "shuffle": True,
    "solver": "adam",
    "tol": 0.0001,
    "validation_fraction": 0.1,
    "verbose": False,
    "warm_start": False,
}
"""
mlp_grid = {
    "activation": ["relu", "identity", "logistic", "tanh"],
    "alpha": [0.0001, 0.001, 0.01, 0.05],
    "batch_size": ["auto"],
    "beta_1": [0.9],
    "beta_2": [0.999],
    "early_stopping": [False, True],
    "epsilon": [1e-08],
    "hidden_layer_sizes": [(100), (150, 100, 50), (120, 80, 40), (100, 50, 30)],
    "max_iter": max_iter,
    "solver": ["adam", "sgd", "lbfgs"],
    "learning_rate": ["constant", "adaptive", "invscaling"],
    "learning_rate_init": [0.001],
    "max_fun": [15000, 20000],
    "momentum": [0.9, 0.99],
    "n_iter_no_change": [10, 20],
    "nesterovs_momentum": [True, False],
    "power_t": [0.5, 0.6],
    "shuffle": [True, False],
    "tol": tol,
    "validation_fraction": [0.1],
    "verbose": [0],
}

# Params: KerasClassifier
"""
"""
# learning algorithm parameters
lr = [1e-2, 1e-3, 1e-4]
decay = [1e-6, 1e-9, 0]

# activation
activation = ["relu", "sigmoid"]

# numbers of layers
nl1 = [0, 1, 2, 3]
nl2 = [0, 1, 2, 3]
nl3 = [0, 1, 2, 3]

# neurons in each layer
nn1 = [
    300,
    700,
    1400,
    2100,
]
nn2 = [100, 400, 800]
nn3 = [50, 150, 300]

# dropout and regularisation
dropout = [0, 0.1, 0.2, 0.3]
l1 = [0, 0.01, 0.003, 0.001, 0.0001]
l2 = [0, 0.01, 0.003, 0.001, 0.0001]

keras_nn_grid = {
    "nl1": nl1,
    "nl2": nl2,
    "nl3": nl3,
    "nn1": nn1,
    "nn2": nn2,
    "nn3": nn3,
    "act": activation,
    "l1": l1,
    "l2": l2,
    "lr": lr,
    "decay": decay,
    "dropout": dropout,
}
# Params: CNNClassifier
"""
{
        model=None,
        *,
        build_fn=None,
        warm_start=False,
        random_state=None
        optimizer='rmsprop',
        loss=None, metrics=None,
        batch_size=None,
        validation_batch_size=None,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        shuffle=True,
        run_eagerly=False,
        epochs=1,
        class_weight=None,
        **kwargs
}
"""
nn_optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name="Adam",
)

epochs = ([10, 200, 100, 300, 400],)

cnn_grid = {
    "validation_split": [0.1],
    "batch_size": [100, 20, 50, 25, 32],
    "epochs": [5, 10],
    "optimizer": ["rmsprop", "adam", nn_optimizer],
    "loss": ["categorical_crossentropy"],
    "metrics": ["accuracy", "f1_score"],
}

# Params: DNNClassifier
"""
{
        model=None,
        *,
        build_fn=None,
        warm_start=False,
        random_state=None
        optimizer='rmsprop',
        loss=None, metrics=None,
        batch_size=None,
        validation_batch_size=None,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        shuffle=True,
        run_eagerly=False,
        epochs=1,
        class_weight=None,
        **kwargs
}
'epochs':[10, 200, 100, 300, 400],
"""
dnn_grid = {
    "validation_split": [0.1],
    "batch_size": [100, 20, 50, 25, 32],
    "epochs": [5, 10],
    "optimizer": ["rmsprop", "adam", nn_optimizer],
    "loss": ["categorical_crossentropy"],
    "metrics": ["accuracy", "f1_score"],
}
# Params End #

gradient_grid_map = {
    "RandomForestClassifier": rf_gradient_grid,
    "LogisticRegression": lr_gradient_grid,
    "LinearSVC": lsvc_gradient_grid,
    "GaussianNB": gaussian_grid,
    "XGBClassifier": xgb_gradient_grid,
    "GradientBoostingClassifier": gbc_grid,
    "AdaBoostClassifier": ada_grid,
    "DecisionTreeClassifier": dt_grid,
    "KNeighborsClassifier": knn_grid,
    "SVC": svc_grid,
    "MLPClassifier": mlp_grid,
    "KerasClassifier": keras_nn_grid,
    "CNNClassifier": cnn_grid,
    "DNNClassifier": dnn_grid,
}
random_grid_map = {
    "RandomForestClassifier": rf_random_grid,
    "LogisticRegression": lr_random_grid,
    "LinearSVC": lsvc_grid,
    "GaussianNB": gaussian_grid,
    "XGBClassifier": xgb_random_grid,
    "GradientBoostingClassifier": gbc_grid,
    "AdaBoostClassifier": ada_grid,
    "DecisionTreeClassifier": dt_grid,
    "KNeighborsClassifier": knn_grid,
    "SVC": svc_grid,
    "MLPClassifier": mlp_grid,
    "KerasClassifier": keras_nn_grid,
    "CNNClassifier": cnn_grid,
    "DNNClassifier": dnn_grid,
}


def get_params_grid(estimator_name, grid_type):
    grid = {}
    if grid_type == "gradient" or grid_type == "half-gradient":
        if estimator_name in gradient_grid_map:
            grid = gradient_grid_map[estimator_name]
    elif grid_type == "random":
        if estimator_name in random_grid_map:
            grid = random_grid_map[estimator_name]
    return grid
