#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

'''
=============================================================================
The below function forms the matrices X, Y.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir: base directory containing input csv files.
- input_file : name of the file containing fnc matrices used as features and 
               actual age of the corresponding subjects
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - X: Fixed effects design matrix of dimension (subjects x features)
 - Y: The response matrix of dimension (subjects,)
=============================================================================
'''
def form_XYMatrices(input_dir, input_file):
    features = pd.read_csv(os.path.join(input_dir, input_file), header=None)
    X = np.array(features[features.columns[:-1]])

    y = features[features.columns[-1:]]
    y = np.array(y[y.columns])[:, 0]

    return (X, y)

"""
Computes performance evaluation metrics for regression model.
"""
def get_metrics(y_true, y_pred):
    results = {}
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['rmse'] = math.sqrt(results['mse'])
    results['mae'] = mean_absolute_error(y_true, y_pred)

    return results
