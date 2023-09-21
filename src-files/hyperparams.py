import pandas as pd
import numpy as np


n_estimators = [int(x) for x in np.linspace(start = 25, stop = 200, num = 5)]
max_features = ['sqrt', 'log2', None]
max_depth = [int(x) for x in np.linspace(3, 10, num = 5)]
max_depth.append(None)
max_leaf_nodes = [int(x) for x in np.linspace(3, 10, num = 5)]
max_leaf_nodes.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
rf_random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes': max_leaf_nodes,
               'bootstrap': bootstrap}

rf_gradient_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9, None],
    'max_leaf_nodes': [3, 6, 9, None],
}

gradient_grid_map = {
        'RandomForestClassifier': rf_gradient_grid,
        'LogisticRegression': None,
        'LinearSVC': None,
        'GaussianNB': None,
        'XGBClassifier': None,
        'GradientBoostingClassifier': None,
        'AdaBoostClassifier': None,
        'DecisionTreeClassifier': None,
        'KNeighborsClassifier': None,
        'SVC': None,
        'MLPClassifier': None,
        }
random_grid_map = {
        'RandomForestClassifier': rf_random_grid,
        'LogisticRegression': None,
        'LinearSVC': None,
        'GaussianNB': None,
        'XGBClassifier': None,
        'GradientBoostingClassifier': None,
        'AdaBoostClassifier': None,
        'DecisionTreeClassifier': None,
        'KNeighborsClassifier': None,
        'SVC': None,
        'MLPClassifier': None,
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

"""
######################## Params: RandomForestClassifier
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

######################## Params: LogisticRegression
{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1500, 'multi_class': 'multinomial', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}

######################## Params: LinearSVC
{'C': 1.0, 'class_weight': None, 'dual': 'auto', 'fit_intercept': True, 'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 100000, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': None, 'tol': 0.0001, 'verbose': 0}

######################## Params: GaussianNB
{'priors': None, 'var_smoothing': 1e-09}

######################## Params: XGBClassifier
{'objective': 'binary:logistic', 'use_label_encoder': None, 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'feature_types': None, 'gamma': None, 'gpu_id': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': None, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': None, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'n_estimators': 100, 'n_jobs': None, 'num_parallel_tree': None, 'predictor': None, 'random_state': None, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}

######################## Params: GradientBoostingClassifier
{'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 1.0, 'loss': 'log_loss', 'max_depth': 1, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'random_state': 0, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

######################## Params: AdaBoostClassifier
{'algorithm': 'SAMME.R', 'base_estimator': 'deprecated', 'estimator': None, 'learning_rate': 1.0, 'n_estimators': 50, 'random_state': None}

######################## Params: DecisionTreeClassifier
{'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'mweight_fraction_leaf': 0.0, 'random_state': 0, 'splitter': 'best'}

######################## Params: KNeighborsClassifier
{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 3, 'p': 2, 'weights': 'uniform'}

######################## Params: SVC
{'C': 1, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': False, 'dom_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}

######################## Params: MLPClassifier

{'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 100, 'momentum': 0.9, 'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 1, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}

"""
