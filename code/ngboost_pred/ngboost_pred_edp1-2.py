import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
import joblib
import os
import random

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. Load preprocessed data
# -----------------------------
data = sio.loadmat("../../data/data_EDP1-2.mat")
X_scaled = data["X_scaled"]   # standardized features
y = data["y"].ravel()         # flatten target to 1D

# -----------------------------
# 2. Split train and test sets
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# -----------------------------
# 3. Train and evaluate NGBoost
# -----------------------------
def train_and_evaluate_ngboost(dist_type, learning_rate, n_estimators, minibatch_frac):
    # Initialize NGBoost model
    ngb = NGBRegressor(
        Dist=dist_type,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        minibatch_frac=minibatch_frac,
        random_state=42
    )

    # Fit model on training data
    ngb.fit(X_train_full, y_train_full)

    # -----------------------------
    # Test set evaluation
    # -----------------------------
    y_pred_test = ngb.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)

    # -----------------------------
    # Train set evaluation
    # -----------------------------
    y_pred_train = ngb.predict(X_train_full)
    train_r2 = r2_score(y_train_full, y_pred_train)
    train_mse = mean_squared_error(y_train_full, y_pred_train)

    # Print evaluation results
    print("="*50)
    print(f"Distribution: {dist_type.__name__}")
    print(f"Best Parameters -> learning_rate: {learning_rate}, "
          f"n_estimators: {n_estimators}, minibatch_frac: {minibatch_frac}")
    print(f"Train R²: {train_r2:.4f}, Train MSE: {train_mse:.4f}")
    print(f"Test  R²: {test_r2:.4f}, Test  MSE: {test_mse:.4f}")
    print("="*50)


# -----------------------------
# 4. Load optimal parameters and run training
# -----------------------------
distribution_param_files = {
    'Normal': '../best_model/model_optimization_results_ngboost_normal_EDP1-2.mat',
    'LogNormal': '../best_model/model_optimization_results_ngboost_lognormal_EDP1-2.mat'
}

for dist in [Normal, LogNormal]:
    dist_name = dist.__name__
    mat_file = sio.loadmat(distribution_param_files[dist_name])
    best_params = mat_file['best_params_ngboost']

    learning_rate = best_params[0, 0]
    n_estimators = int(best_params[0, 1])
    minibatch_frac = best_params[0, 2]

    # Train and evaluate NGBoost with the best parameters
    train_and_evaluate_ngboost(dist, learning_rate, n_estimators, minibatch_frac)
