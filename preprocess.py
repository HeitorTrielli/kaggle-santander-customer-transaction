from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_santander_data():
    return [
        pd.read_csv(Path("santander-customer-transaction-prediction/train.csv")),
        pd.read_csv(Path("santander-customer-transaction-prediction/test.csv")),
    ]


train, test = load_santander_data()


def preprocess_feats(feats: pd.DataFrame):
    feats = feats.set_index("ID_code")

    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    num_pipeline = Pipeline([("imputer", num_imputer), ("scaler", scaler)])

    preprocessed_feats = num_pipeline.fit_transform(feats)

    return preprocessed_feats


feats = train.drop(columns="target")

processed_feats = preprocess_feats(train.drop(columns="target"))
