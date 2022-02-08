import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import classification_report
import mlflow
import logging
import sys
import warnings
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

df_clean = pd.read_csv("../Data/clean_dataset.csv")

"""
    Our dataset is imbalanced. We'll try to solve this by Oversampling or
    generating synthetic samples. 
"""


def model_build(hp):
    model = Sequential()
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, input_dim=20, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])
    return model


def trainer(X_train, Y_train, X_test, Y_test):
    tuner = kt.Hyperband(model_build,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='ml_tuning')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, Y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, Y_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, Y_train, epochs=best_epoch, validation_split=0.2)

    _, accuracy = hypermodel.evaluate(X_train, Y_train)
    print('Accuracy: %.2f' % (accuracy * 100))

    y_pred = hypermodel.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(Y_test, y_pred_bool))
    mlflow.log_param('f1_score', f1_score(Y_test, y_pred_bool, average='macro'))
    return hypermodel


def mode(input_mode=2):
    if input_mode == 1:
        X_df = df_clean.drop('Sale', axis=1)
        y_df = df_clean.Sale
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=27)
        X_df = pd.concat([X_tr, y_tr], axis=1)
        X_VAL = pd.concat([X_te, y_te], axis=1)
        clicked = X_df[X_df.Sale == 1]
        nonclick = X_df[X_df.Sale == 0]
        clicked_upsampling = resample(clicked,
                                      replace=True,  # sample with replacement
                                      n_samples=len(nonclick),  # match number in majority class
                                      random_state=27)  # reproducible results
        upsampled = pd.concat([nonclick, clicked_upsampling])
    else:
        X_df = df_clean.drop('Sale', axis=1)
        y_df = df_clean.Sale
        X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_df, test_size=0.2, random_state=27)
        X_VAL = pd.concat([X_te, y_te], axis=1)

        sm = SMOTE(random_state=27)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        upsampled = pd.concat([X_tr, y_tr], axis=1)

    """
        We will then perform PCA on data since it's too large to feed into our network.
        We kept 20 components.
    """
    y_train_label = upsampled['Sale'].to_numpy()
    X = upsampled.drop('Sale', axis=1).to_numpy()
    PCA_N_COMP = 20
    pca = PCA(n_components=PCA_N_COMP)
    pca_projection = pca.fit(X)
    compact_data_train = pca_projection.transform(X)

    y_test_label = X_VAL['Sale'].to_numpy()
    X = X_VAL.drop('Sale', axis=1).to_numpy()
    PCA_N_COMP = 20
    pca = PCA(n_components=PCA_N_COMP)
    pca_projection = pca.fit(X)
    compact_data_test = pca_projection.transform(X)

    return trainer(compact_data_train, y_train_label, compact_data_test, y_test_label)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    with mlflow.start_run() as runner:
        model_path = os.path.join('mlflow_models', "models_" + runner.info.run_id)
        model = mode(2)
        mlflow.keras.save_model(path=model_path, python_model=model)
        reload_model = mlflow.pyfunc.load_model(model_path)
        print(f'runner id is: {runner.info.run_id}')
