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
n_estimators = [int(x) for x in np.linspace(start=25, stop=200, num=5)]
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
}

rf_gradient_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [3, 6, 9, None],
    "max_leaf_nodes": [3, 6, 9, None],
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
penalty = ["l2"]
c_values = [100, 10, 1.0, 0.1, 0.01]
c_values_random = [int(x) for x in np.linspace(start=1, stop=200, num=10)]

lr_gradient_grid = {
    "penalty": penalty,
    "solver": solvers,
    "C": c_values,
    "max_iter": [50, 100, 150, 200, 1500],
}
lr_random_grid = {
    "penalty": penalty,
    "solver": solvers,
    "C": c_values_random,
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
    "max_iter": 100000,
    "multi_class": "ovr",
    "penalty": "l2",
    "random_state": None,
    "tol": 0.0001,
    "verbose": 0,
}
"""
gamma_random = [int(x) for x in np.linspace(start=1, stop=1000, num=10)]

lsvc_gradient_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "max_iter": [50, 100, 150, 200, 1500],
}
lsvc_random_grid = {
    "C": c_values_random,
    "max_iter": [50, 100, 150, 200, 1500],
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
    "dom_state": None,
    "shrinking": True,
    "tol": 0.001,
    "verbose": False,
}
"""

svc_gradient_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "max_iter": [50, 100, 150, 200, 1500],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
}
svc_random_grid = {
    "C": c_values_random,
    "max_iter": [50, 100, 150, 200, 1500],
    "gamma": gamma_random,
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
}
# Params: GaussianNB
gaussian_grid = {
    "priors": None,
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
}

xgb_random_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "gamma": gamma_random,
    "reg_alpha": [0, 0.5, 1],
    "reg_lambda": [0.5, 1, 0.75],
    "colsample_bytree": [0.5, 1, 0.75],
    "min_child_weight": min_child_weights,
    "learning_rate": [1, 0.5, 0.25, 0.1, 0.05, 0.01],
    "booster": ["gbtree", "gblinear", "dart"],
    "base_score": [0.2, 0.5, 0.75],
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
    "subsample": [0.8],
    "learning_rate": learning_rates,
}
gbc_random_grid = {
    "n_estimators": n_estimators,
    "learning_rate": learning_rates,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": max_leaf_nodes,
    "subsample": [0.8],
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
    "learning_rate": learning_rates,
    "n_estimators": n_estimators,
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
    "mweight_fraction_leaf": 0.0,
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
}
dt_random_grid = {
    "max_features": max_features,
    "max_depth": max_depth,
    "max_leaf_nodes": max_leaf_nodes,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
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
}
knn_random_params = {
    "n_neighbors": n_neighbors,
    "leaf_size": leaf_size,
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "metric": ["minkowski", "euclidean", "manhattan"],
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
    "hidden_layer_sizes": [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
    "max_iter": [50, 100, 150, 200, 1500],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["sgd", "adam", "lbfgs"],
    "alpha": [0.0001, 0.001, 0.01, 0.05],
    "learning_rate": ["constant", "adaptive", "invscaling"],
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
