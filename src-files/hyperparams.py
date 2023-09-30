"""
"""
import numpy as np

# import packages for hyperparameters tuning
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Params Start #

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
n_estimators = [int(x) for x in np.linspace(start=25, stop=500, num=10)]
max_features = ["sqrt", "log2", None]
max_depth = [int(x) for x in np.linspace(3, 10, num=5)]
max_depth.append(None)
max_leaf_nodes = [int(x) for x in np.linspace(3, 10, num=5)]
max_leaf_nodes.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
min_child_weights = [int(x) for x in np.linspace(3, 20, num=10)]
bootstrap = [True, False]
rf_random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": max_leaf_nodes,
    "bootstrap": bootstrap,
    "class_weight": ["balanced", "balanced_subsample", None],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_samples": [1, 2, 5, None],
    "min_impurity_decrease": [0.0, 0.1],
    "min_weight_fraction_leaf": [0.0, 0.1],
    "n_jobs": [-1, None],
    "oob_score": [False, True],
    "random_state": [None],
    "verbose": [0],
    "warm_start": [False, True],
}

rf_gradient_grid = {
    "n_estimators": [50, 100, 150, 200, 300, 500],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [3, 6, 9, None],
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": [3, 6, 9, None],
    "bootstrap": bootstrap,
    "class_weight": ["balanced", "balanced_subsample", None],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_samples": [1, 2, 5, None],
    "min_impurity_decrease": [0.0, 0.1],
    "min_weight_fraction_leaf": [0.0, 0.1],
    "n_jobs": [-1, None],
    "oob_score": [False, True],
    "random_state": [None],
    "verbose": [0],
    "warm_start": [False, True],
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

solvers = ["newton-cg", "lbfgs", "liblinear"]
penalty = ["l1", "l2", "elasticnet", None]
c_values = [100, 10, 1.0, 0.1, 0.01]

lr_gradient_grid = {
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": solvers,
    "C": c_values,
    "class_weight": [None],
    "dual": [False, True],
    "fit_intercept": [False, True],
    "intercept_scaling": [1],
    "l1_ratio": [0.0, 0.5, 1.0, None],
    "n_jobs": [-1, None],
    "random_state": [None],
    "tol": [0.0001, 0.0005, 0.001],
    "verbose": [0],
    "warm_start": [False, True],
    "multi_class": ["auto", "ovr", "multinomial"],
    "max_iter": [50, 100, 150, 200, 1500],
}
lr_random_grid = {
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": solvers,
    "C": c_values,
    "class_weight": [None],
    "dual": [False, True],
    "fit_intercept": [False, True],
    "intercept_scaling": [1],
    "l1_ratio": [0.0, 0.5, 1.0, None],
    "n_jobs": [-1, None],
    "random_state": [None],
    "tol": [0.0001, 0.0005, 0.001],
    "verbose": [0],
    "warm_start": [False, True],
    "multi_class": ["auto", "ovr", "multinomial"],
    "max_iter": [50, 100, 150, 200, 1500],
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
    "max_iter": [50, 100, 150, 200, 1500],
    "multi_class": "ovr",
    "penalty": "l2",
    "random_state": None,
    "tol": 0.0001,
    "verbose": 0,
}
"""

lsvc_gradient_grid = {
    "C": c_values,
    "max_iter": [50, 100, 150, 200, 1500, 100000],
    "class_weight": [None],
    "dual": [False, True],
    "fit_intercept": [False, True],
    "intercept_scaling": [1],
    "loss": ["squared_hinge", "hinge"],
    "multi_class": ["ovr", "crammer_singer"],
    "penalty": ["l1", "l2"],
    "random_state": [None],
    "tol": [0.0001, 0.0005, 0.001],
    "verbose": [0],
}
lsvc_random_grid = {
    "C": c_values,
    "max_iter": [50, 100, 150, 200, 1500, 100000],
    "class_weight": [None],
    "dual": [False, True],
    "fit_intercept": [False, True],
    "intercept_scaling": [1],
    "loss": ["squared_hinge", "hinge"],
    "multi_class": ["ovr", "crammer_singer"],
    "penalty": ["l1", "l2"],
    "random_state": [None],
    "tol": [0.0001, 0.0005, 0.001],
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

svc_gradient_grid = {
    "C": c_values,
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "break_ties": [False, True],
    "cache_size": [200, 500],
    "class_weight": [None],
    "coef0": [0.0, 0.1],
    "decision_function_shape": ["ovr", "ovo"],
    "degree": [3],
    "gamma": ["scale", "auto"],
    "max_iter": [50, 100, 150, 200, 1500, 100000, -1],
    "probability": [False, True],
    "random_state": [None],
    "shrinking": [True, False],
    "tol": [0.0001, 0.0005, 0.001],
    "verbose": [0],
}
svc_random_grid = {
    "C": c_values,
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "break_ties": [False, True],
    "cache_size": [200, 500],
    "class_weight": [None],
    "coef0": [0.0, 0.1],
    "decision_function_shape": ["ovr", "ovo"],
    "degree": [3],
    "gamma": ["scale", "auto"],
    "max_iter": [50, 100, 150, 200, 1500, 100000, -1],
    "probability": [False, True],
    "random_state": [None],
    "shrinking": [True, False],
    "tol": [0.0001, 0.0005, 0.001],
    "verbose": [0],
}
# Params: GaussianNB
gaussian_grid = {
    "var_smoothing": np.logspace(0, -9, num=100),
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
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 6, 9, None],
    "learning_rate": [1, 0.5, 0.25, 0.1, 0.05, 0.01],
    "gamma": [0, 0.5, 0.75],
    "reg_alpha": [0, 0.5, 1],
    "reg_lambda": [0.5, 1, 0.75],
    "base_score": [0.2, 0.5, 0.75],
    "eta": [0.2, 0.3, 0.5],
    "colsample_bytree": [0.5, 1, 0.75],
    "booster": ["gbtree", "gblinear", "dart"],
    "verbosity": [0, None],
    "objective": ["binary:logistic", "reg:logistic", "binary:logitraw", "binary:hinge"],
    "n_jobs": [-1, None],
}

xgb_random_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "gamma": [0, 0.5, 0.75],
    "reg_alpha": [0, 0.5, 1],
    "reg_lambda": [0.5, 1, 0.75],
    "colsample_bytree": [0.5, 1, 0.75],
    "min_child_weight": min_child_weights,
    "learning_rate": [1, 0.5, 0.25, 0.1, 0.05, 0.01],
    "booster": ["gbtree", "gblinear", "dart"],
    "base_score": [0.2, 0.5, 0.75],
    "objective": ["binary:logistic", "reg:logistic", "binary:logitraw", "binary:hinge"],
    "n_jobs": [-1, None],
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
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]

gbc_gradient_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [3, 6, 9, None],
    "max_leaf_nodes": [3, 6, 9, None],
    "subsample": [0.8, 1.0],
    "learning_rate": learning_rates,
    "ccp_alpha": [0.0, 0.01, 0.5],
    "criterion": ["friedman_mse", "squared_error"],
    "min_impurity_decrease": [0.0, 0.1],
    "min_samples_leaf": min_samples_leaf,
    "min_samples_split": min_samples_split,
    "min_weight_fraction_leaf": [0.0, 0.1],
    "tol": [0.0001, 0.0005, 0.001],
    "validation_fraction": [0.1],
    "verbose": [0],
    "warm_start": [False, True],
}
gbc_random_grid = {
    "n_estimators": n_estimators,
    "learning_rate": learning_rates,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": max_leaf_nodes,
    "subsample": [0.8, 1.0],
    "ccp_alpha": [0.0, 0.01, 0.5],
    "criterion": ["friedman_mse", "squared_error"],
    "min_impurity_decrease": [0.0, 0.1],
    "min_weight_fraction_leaf": [0.0, 0.1],
    "tol": [0.0001, 0.0005, 0.001],
    "validation_fraction": [0.1],
    "verbose": [0],
    "warm_start": [False, True],
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
ada_gradient_grid = {
    "algorithm": ["SAMME", "SAMME.R"],
    "n_estimators": [50, 100, 150, 200],
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
dt_gradient_grid = {
    "max_features": ["sqrt", "log2", None],
    "max_depth": [3, 6, 9, None],
    "max_leaf_nodes": [3, 6, 9, None],
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "ccp_alpha": [0.0, 0.01, 0.5],
    "class_weight": [None],
    "min_impurity_decrease": [0.0, 0.1],
    "min_weight_fraction_leaf": [0.0, 0.1],
    "splitter": ["best", "random"],
}
dt_random_grid = {
    "max_features": max_features,
    "ccp_alpha": [0.0, 0.01, 0.5],
    "max_depth": max_depth,
    "max_leaf_nodes": max_leaf_nodes,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "class_weight": [None],
    "min_impurity_decrease": [0.0, 0.1],
    "min_weight_fraction_leaf": [0.0, 0.1],
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
n_neighbors = [int(x) for x in np.linspace(start=3, stop=20, num=8)]
leaf_size = [int(x) for x in np.linspace(start=30, stop=50, num=3)]
knn_gradient_params = {
    "n_neighbors": [5, 7, 9, 11, 13, 15],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "metric": ["minkowski", "euclidean", "manhattan"],
    "leaf_size": [30, 40, 50],
    "n_jobs": [-1, None],
    "p": [2, 1],
}
knn_random_params = {
    "n_neighbors": [5, 7, 9, 11, 13, 15],
    "leaf_size": [30, 40, 50],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "metric": ["minkowski", "euclidean", "manhattan"],
    "n_jobs": [-1, None],
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
mlp_gradient_params = {
    "activation": ["identity", "logistic", "tanh", "relu"],
    "alpha": [0.0001, 0.001, 0.01, 0.05],
    "batch_size": ["auto"],
    "beta_1": [0.9],
    "beta_2": [0.999],
    "early_stopping": [False, True],
    "epsilon": [1e-08],
    "hidden_layer_sizes": [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
    "max_iter": [50, 100, 150, 200, 1500, 2000],
    "solver": ["sgd", "adam", "lbfgs"],
    "learning_rate": ["constant", "adaptive", "invscaling"],
    "learning_rate_init": [0.001],
    "max_fun": [15000, 20000],
    "momentum": [0.9, 0.99],
    "n_iter_no_change": [10, 20],
    "nesterovs_momentum": [True, False],
    "power_t": [0.5, 0.6],
    "shuffle": [True, False],
    "tol": [0.0001, 0.0005, 0.001],
    "validation_fraction": [0.1],
    "verbose": [0],
    "warm_start": [False, True],
}

# Params End #

gradient_grid_map = {
    "RandomForestClassifier": rf_gradient_grid,
    "LogisticRegression": lr_gradient_grid,
    "LinearSVC": lsvc_gradient_grid,
    "GaussianNB": gaussian_grid,
    "XGBClassifier": xgb_gradient_grid,
    "GradientBoostingClassifier": gbc_gradient_grid,
    "AdaBoostClassifier": ada_gradient_grid,
    "DecisionTreeClassifier": dt_gradient_grid,
    "KNeighborsClassifier": knn_gradient_params,
    "SVC": svc_gradient_grid,
    "MLPClassifier": mlp_gradient_params,
}
random_grid_map = {
    "RandomForestClassifier": rf_random_grid,
    "LogisticRegression": lr_random_grid,
    "LinearSVC": lsvc_random_grid,
    "GaussianNB": gaussian_grid,
    "XGBClassifier": xgb_random_grid,
    "GradientBoostingClassifier": gbc_random_grid,
    "AdaBoostClassifier": ada_gradient_grid,
    "DecisionTreeClassifier": dt_gradient_grid,
    "KNeighborsClassifier": knn_random_params,
    "SVC": svc_random_grid,
    "MLPClassifier": mlp_gradient_params,
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
