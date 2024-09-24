import os
import time

import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from preprocess import load_santander_data, preprocess_feats

santander_train, santander_test = load_santander_data()

raw_train_feats = santander_train.drop(columns="target")
raw_test_feats = santander_test

train_label = santander_train.target

train_feats = preprocess_feats(raw_train_feats)
test_feats = preprocess_feats(raw_test_feats)


def model_learning_rate():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=train_feats.shape[1:]),
            tf.keras.layers.Flatten(input_shape=[28, 28]),
            tf.keras.layers.Dense(
                100, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                100, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                100, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                100, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                100, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["AUC"],
    )

    return model


model = model_learning_rate()

t0 = time.time()
n_epochs = 30
history = model.fit(train_feats, train_label, epochs=n_epochs, validation_split=0.15)
t1 = time.time()

prediction = model.predict(test_feats)

df = pd.concat(
    [
        raw_test_feats[["ID_code"]],
        pd.DataFrame(prediction, columns=["target"]),
    ],
    axis=1,
)

df.to_csv("keras_submission.csv", index=False)
