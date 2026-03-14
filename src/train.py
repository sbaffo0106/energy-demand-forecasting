from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def make_Xy(df, target_col="PJME_MW"):
    """
    Split dataframe into:
    X = features (all columns except target)
    y = target (target column)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def make_train_test_split(X, y, test_size=0.2):
    """
    Temporal train/test split for time series forecasting.
    Keeps chronological order.
    Returns: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, shuffle=False)

def build_ridge_pipeline(alpha=1.0):
    """
    Build a Ridge regression pipeline with feature scaling.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha))
    ])
    return pipe

def build_random_forest_model(n_estimators=200, 
                              max_depth=None, 
                              random_state=42):
    """
    Build a Random Forest regression model.
    """
    return RandomForestRegressor(n_estimators=n_estimators, 
                                 max_depth=max_depth,
                                 random_state=random_state
    )

def build_gradient_boosting_model(n_estimators=200, 
                                  learning_rate=0.05,
                                  max_depth=3,
                                  random_state=42):
    """
    Build a Gradient Boosting regression model.
    """
    return GradientBoostingRegressor(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth,
                                     random_state=random_state)

def fit_predict(estimator, X_train, y_train, X_test):
    """
    Fit an estimator (model or pipeline) on the training data 
    and generate predictions on the test set.
    """
    estimator.fit(X_train, y_train)
    return estimator.predict(X_test)
