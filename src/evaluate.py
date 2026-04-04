import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
from sklearn.base import clone


from sklearn.base import clone
import numpy as np
import time

from sklearn.base import clone
import numpy as np

from sklearn.base import clone
import numpy as np

def rolling_forecast(model, X, y, initial_train_size, step=1, window_size=5000):
    """
    Sliding-window rolling forecast evaluation.

    Parameters:
    - model: sklearn-like estimator
    - X, y: full dataset (pandas DataFrame/Series)
    - initial_train_size: starting index for evaluation
    - step: forecast horizon at each iteration
    - window_size: fixed size of training window (sliding window)
    """

    y_true = []
    y_pred = []

    for i in range(initial_train_size, len(X), step):

        start = max(0, i - window_size)

        X_train = X.iloc[start:i]
        y_train = y.iloc[start:i]

        X_test = X.iloc[i:i + step]
        y_test = y.iloc[i:i + step]

        if len(X_test) == 0:
            break

        m = clone(model)
        m.fit(X_train, y_train)

        preds = m.predict(X_test)

        y_true.extend(y_test.values)
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def evaluate_regression(y_true, y_pred):
    """
    Compute standard regression metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    