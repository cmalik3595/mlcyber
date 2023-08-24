"""
"""
import warnings

import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn import preprocessing, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action="ignore", category=FutureWarning)

KF = KFold(random_state=42, shuffle=True)


def model_plot(prob, y_test, model_name, i):

    # Calculate  ROC curves
    fpr, tpr, thresholds = roc_curve(y_test, prob)

    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"{model_name}_{i} ROC Curve (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    fig_roc_file_save = f"./output/model-comparison/plots/{model_name}_{i}_roc.html"
    fig.write_html(file=fig_roc_file_save, include_plotlyjs="cdn")

    return


def perf_table(model_names, predictions, y_test, i):

    # Build preliminary results table
    pca_performance_cols = ["Brier Score", "Precision", "Recall", "F1"]
    pca_performance = pd.DataFrame(
        columns=pca_performance_cols,
    )

    for name, pred in zip(model_names, predictions):
        name_brier = brier_score_loss(y_test, pred)
        name_prec = precision_score(y_test, pred, average="weighted")
        name_recall = recall_score(y_test, pred, average="weighted")
        name_f1 = f1_score(y_test, pred, average="weighted")

        pca_performance.loc[name] = pd.Series(
            {
                "Brier Score": name_brier,
                "Precision": name_prec,
                "Recall": name_recall,
                "F1": name_f1,
            }
        )

    perf_file_save = f"./output/model-comparison/pca_performance_table_{i}.html"
    with open(perf_file_save, "w") as html_open:
        pca_performance.sort_index().to_html(html_open, escape=False)

    return


def build_models(X: pd.DataFrame, y: pd.Series, j) -> None:
    df = X
    df[y.name] = y
    df = df.dropna()
    X = df.drop(y.name, axis=1)
    svc_scores = []
    rfc_scores = []
    for i, split in enumerate(KF.split(X)):
        train_index, test_index = split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Some processing
        normalizer = preprocessing.Normalizer(norm="l2")
        X_train_norm = normalizer.fit_transform(X_train)
        X_test_norm = normalizer.fit_transform(X_test)
        svc = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", LinearSVC(max_iter=64000)),
            ]
        )
        svc.fit(X_train, y_train)
        svc_score = svc.score(X_test, y_test)
        svc_scores.append(svc_score)
        svc_preds = svc.predict(X_test_norm)
        model_name = "SVC"
        model_plot(svc_preds, y_test, model_name, i)
        print("SVC:\n", classification_report(y_test, svc_preds))
        print(f"SVC {i:02}", round(svc_score, 2))

        rfc = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rfc", RandomForestClassifier()),
            ]
        )
        rfc.fit(X_train, y_train)
        rfc_score = rfc.score(X_test, y_test)
        rfc_scores.append(rfc_score)
        rfc_preds = rfc.predict(X_test_norm)
        model_name = "RFC"
        model_plot(rfc_preds, y_test, model_name, i)
        print("RFC:\n", classification_report(y_test, rfc_preds))

        print(f"RFC {i:02}", round(rfc_score, 2))

        # Let's try logistic regression
        log_reg = LogisticRegression(max_iter=300, fit_intercept=True)
        log_reg_fit = log_reg.fit(X_train_norm, y_train)
        log_preds = log_reg_fit.predict(X_test_norm)
        print("Logistic:\n", classification_report(y_test, log_preds))

        # Fit random forest model
        rf_model = RandomForestClassifier(oob_score=True, random_state=1234)
        rf_model.fit(X_train_norm, y_train)

        rf_preds = rf_model.predict(X_test_norm)

        print("Random Forest:\n", classification_report(y_test, rf_preds))

        # RF ROC plot
        model_name = "Random Forest"
        rf_probs = rf_model.predict_proba(X_test_norm)
        prob = rf_probs[:, 1]
        model_plot(prob, y_test, model_name, i)

        model_name = "Logistic"
        log_probs = log_reg.predict_proba(X_test_norm)
        prob = log_probs[:, 1]
        model_plot(prob, y_test, model_name, i)

        # SVM
        svm_model = svm.SVC(probability=True)
        svm_fitted = svm_model.fit(X_train_norm, y_train)
        svm_preds = svm_fitted.predict(X_test_norm)

        print("SVM:\n", classification_report(y_test, svm_preds))

        # SVM ROC plot
        model_name = "SVM"
        svm_probs = svm_model.predict_proba(X_test_norm)
        prob = svm_probs[:, 1]
        model_plot(prob, y_test, model_name, i)

        # KNN
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_fitted = knn_model.fit(X_train_norm, y_train)
        knn_preds = knn_fitted.predict(X_test_norm)

        print("KNN:\n", classification_report(y_test, knn_preds))

        # KNN ROC plot
        model_name = "K-Nearest Neighbor"
        knn_probs = knn_model.predict_proba(X_test_norm)
        prob = knn_probs[:, 1]
        model_plot(prob, y_test, model_name, i)

        # Decision tree classifier
        dtc_model = DecisionTreeClassifier(random_state=1234)
        dtc_fitted = dtc_model.fit(X_train_norm, y_train)
        dtc_preds = dtc_fitted.predict(X_test_norm)

        print("Decision tree classifier:\n", classification_report(y_test, dtc_preds))

        # Decision Tree Classifier ROC plot
        model_name = "Decision Tree Classifier"
        dtc_probs = dtc_model.predict_proba(X_test_norm)
        prob = dtc_probs[:, 1]
        model_plot(prob, y_test, model_name, i)

        # Linear discriminant analysis
        lda_model = LinearDiscriminantAnalysis()
        lda_fitted = lda_model.fit(X_train_norm, y_train)
        lda_preds = lda_fitted.predict(X_test_norm)

        print(
            "Linear discriminant analysis:\n", classification_report(y_test, lda_preds)
        )

        # Linear Discriminant Analysis ROC plot
        model_name = "Linear Discriminant Analysis"
        lda_probs = lda_model.predict_proba(X_test_norm)
        prob = lda_probs[:, 1]
        model_plot(prob, y_test, model_name, i)

        # Gaussian Naive Bayes
        gnb_model = GaussianNB()
        gnb_fitted = gnb_model.fit(X_train_norm, y_train)
        gnb_preds = gnb_fitted.predict(X_test_norm)

        print("Gaussian Naive Bayes:\n", classification_report(y_test, gnb_preds))

        # Gaussian Naive Bayes ROC plot
        model_name = "Gaussian Naive Bayes"
        gnb_probs = gnb_model.predict_proba(X_test_norm)
        prob = gnb_probs[:, 1]
        model_plot(prob, y_test, model_name, i)

        # XGBoost
        xg_model = xgb.XGBClassifier(
            tree_method="approx",
            predictor="cpu_predictor",
            verbosity=1,
            eval_metric=["merror", "map", "auc"],
            objective="binary:logistic",
            eta=0.3,
            n_estimators=100,
            colsample_bytree=0.95,
            max_depth=3,
            reg_alpha=0.001,
            reg_lambda=150,
            subsample=0.8,
        )
        xgb_model = xg_model.fit(X_train_norm, y_train)
        xgb_preds = xgb_model.predict(X_test_norm)

        print("XGBoost:\n", classification_report(y_test, xgb_preds))

        # XGB ROC plot
        model_name = "XGBoost"
        xgb_probs = xgb_model.predict_proba(X_test_norm)
        prob = xgb_probs[:, 1]
        model_plot(prob, y_test, model_name, i)
        print("********************************************\n")

    # Create performance table
    model_names = [
        "Random Forest",
        "Logistic",
        "SVM",
        "KNN",
        "Decision Trees",
        "LDA",
        "Gaussian Naive Bayes",
        "XGBoost",
        "RFC",
        "SVC",
    ]
    predictions = [
        rf_preds,
        log_preds,
        svm_preds,
        knn_preds,
        dtc_preds,
        lda_preds,
        gnb_preds,
        xgb_preds,
        rfc_preds,
        svc_preds,
    ]

    perf_table(model_names, predictions, y_test, j)

    return None


if __name__ == "__main__":
    pass
