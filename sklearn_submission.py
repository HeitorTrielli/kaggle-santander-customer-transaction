import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from preprocess import load_santander_data, preprocess_feats

santander_train, santander_test = load_santander_data()

raw_train_feats = santander_train.drop(columns="target")
raw_test_feats = santander_test

train_label = santander_train.target

train_feats = preprocess_feats(raw_train_feats)
test_feats = preprocess_feats(raw_test_feats)

# SGD
sgd_clf = SGDClassifier(random_state=27, n_jobs=8)

sgd_param_grid = [
    {
        "alpha": np.arange(1e-5, 1e-3, 2e-4),
        "loss": ["log_loss", "hinge"],
        "penalty": ["l2", "elasticnet"],
    }
]

sgd_cv = GridSearchCV(
    estimator=sgd_clf, param_grid=sgd_param_grid, scoring="roc_auc", verbose=10
)

sgd_cv.fit(train_feats, train_label)
sgd_cv.best_params_
sgd_cv.best_estimator_

sgd_results = pd.DataFrame(sgd_cv.cv_results_)
sgd_results.sort_values("mean_test_score", ascending=False)

raw_test_feats["target"] = sgd_cv.best_estimator_.predict(test_feats)
sgd_submission = raw_test_feats[["ID_code", "target"]]

sgd_submission.to_csv("sgd_submission.csv", index=False)

# KNN
knn_clf = KNeighborsClassifier(n_jobs=8)

knn_param_grid = [
    {
        "n_neighbors": range(1, 100, 10),
        "weights": ["uniform", "distance"],
    }
]

knn_cv = GridSearchCV(
    estimator=knn_clf, param_grid=knn_param_grid, scoring="roc_auc", verbose=10
)

knn_cv.fit(train_feats, train_label)
knn_cv.best_params_
knn_cv.best_estimator_

knn_results = pd.DataFrame(knn_cv.cv_results_)
knn_results.sort_values("mean_test_score", ascending=False)


raw_test_feats["target"] = knn_cv.best_estimator_.predict(test_feats)
knn_submission = raw_test_feats[["ID_code", "target"]].rename(
    columns={"ID_code": "ID_code"}
)

knn_submission.to_csv("knn_submission.csv", index=False)


# Random Forest
rf_clf = RandomForestClassifier(random_state=42, n_jobs=8)

rf_param_grid = [
    {
        "n_estimators": [100, 150, 250],
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": np.arange(0.1, 0.4, 0.1),
        "oob_score": [True],
    }
]

rf_cv = GridSearchCV(
    estimator=rf_clf, param_grid=rf_param_grid, scoring="roc_auc", verbose=10, cv=3
)

rf_cv.fit(train_feats, train_label)
rf_cv.best_params_
rf_cv.best_estimator_

rf_results = pd.DataFrame(rf_cv.cv_results_)
rf_results.sort_values("mean_test_score", ascending=False)

raw_test_feats["target"] = rf_cv.best_estimator_.predict(test_feats)
rf_submission = raw_test_feats[["ID_code", "target"]].rename(
    columns={"ID_code": "ID_code"}
)

rf_submission.to_csv("rf_submission.csv", index=False)
