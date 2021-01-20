import json
import math

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from svr_utils import form_XYMatrices

"""
This code aggregates data from all the local sites and performs regression on the combined data.
"""

"""
Performs LinearSVR on the passed data.
"""


def aggregated_SVR(X_train, X_test, y_train, y_test):
    input_list = None
    for indx, conf in enumerate(json.loads(open('../test/inputspec.json').read())):
        if input_list is None:
            conf.pop("input_train_csv_path")
            conf.pop("input_test_csv_path")
            input_list = conf
            break

    # X = X.astype(np.double)
    regr = make_pipeline(preprocessing.MinMaxScaler(),
                         LinearSVR
                         (epsilon=input_list["epsilon_local"]["value"],
                          tol=input_list["tolerance_local"]["value"],
                          C=input_list["regularization_local"]["value"],
                          loss=input_list["loss_local"]["value"],
                          fit_intercept=input_list["fit_intercept_local"]["value"],
                          intercept_scaling=input_list["intercept_scaling_local"]["value"],
                          dual=input_list["dual_local"]["value"],
                          random_state=input_list["random_state_local"]["value"],
                          max_iter=input_list["max_iterations_local"]["value"]))

    regr.fit(X_train, y_train)
    params = regr.get_params()
    svr2 = params['linearsvr']
    w = svr2.coef_
    w = np.squeeze(w)
    intercept_aggr = svr2.intercept_

    y_train_pred = regr.predict(X_train)
    mse_train_combined = mean_squared_error(y_train, y_train_pred)
    rmse_train_combined = math.sqrt(mse_train_combined)
    mae_train_local = mean_absolute_error(y_train, y_train_pred)

    y_test_pred = regr.predict(X_test)
    mse_test_combined = mean_squared_error(y_test, y_test_pred)
    rmse_test_combined = math.sqrt(mse_test_combined)
    mae_test_local = mean_absolute_error(y_test, y_test_pred)

    output_dict = {
        # "intercept_aggregated": intercept_combined.tolist(),
        # "w_aggregated": w.tolist(),
        "n_train_samples_aggregated": len(y_train),
        "n_test_samples_aggregated": len(y_test),
        "rmse_train_aggregated": float(rmse_train_combined),
        "rmse_test_aggregated": float(rmse_test_combined),
        "mae_train_aggregated": float(mae_train_local),
        "mae_test_aggregated": float(mae_test_local),
        "phase": "aggregated",
    }

    print(output_dict)


"""
Combines data from all the local sites.
"""


def combine_all_local_data():
    X_train, y_train, X_test, y_test = None, None, None, None
    for indx, conf in enumerate(json.loads(open('../test/inputspec.json').read())):
        [local_X_train, local_y_train] = form_XYMatrices(
            input_dir=f"../test/input/local{indx}/simulatorRun/{conf['split_type']['value']}/",
            input_file=conf['input_train_csv_path']['value'])
        [local_X_test, local_y_test] = form_XYMatrices(
            input_dir=f"../test/input/local{indx}/simulatorRun/{conf['split_type']['value']}/",
            input_file=conf['input_test_csv_path']['value'])
        if X_train is None:
            X_train = local_X_train
            y_train = local_y_train
            X_test = local_X_test
            y_test = local_y_test
        else:
            X_train = np.vstack((X_train, local_X_train))
            y_train = np.hstack((y_train, local_y_train))
            X_test = np.vstack((X_test, local_X_test))
            y_test = np.hstack((y_test, local_y_test))

    return (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = combine_all_local_data()
    print("Combined train and test data from all the local clients. Running SVR now.")
    aggregated_SVR(X_train, X_test, y_train, y_test)
