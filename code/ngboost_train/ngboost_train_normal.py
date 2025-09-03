import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ssa import SSA   # import the SSA class
import pandas as pd
import random
from sklearn.exceptions import NotFittedError
import warnings

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. Load preprocessed data
# -----------------------------
data = sio.loadmat("../../data/data_EDP1-1.mat")
X = data["X_scaled"]       # standardized features
y = data["y"].ravel()      # flatten target to 1D

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Further split train into train+validation
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)


# -----------------------------
# 4. Define NGBoost fitness function
# -----------------------------
def ngboost_fitness(params):
    """
    Fitness function for SSA optimization of NGBoost.
    params: [learning_rate, n_estimators, minibatch_frac]
    Returns the validation MSE (to be minimized).
    """
    learning_rate, n_estimators, minibatch_frac = params

    # Sanity checks to avoid invalid values
    if n_estimators < 1 or minibatch_frac < 0.01:
        return 1e6  # large penalty score

    try:
        # Initialize NGBoost model
        ngb = NGBRegressor(
            Dist=Normal,
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            minibatch_frac=minibatch_frac,
            random_state=42,
            verbose=False
        )

        # Suppress sklearn warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ngb.fit(X_train, y_train)

        # Predict on validation set
        y_pred = ngb.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)

        # Return a large value if MSE is invalid
        if np.isnan(mse):
            return 1e6
        return mse

    except (ValueError, NotFittedError, Exception) as e:
        print(f"[WARN] Fitness evaluation failed for params: {params}, error: {e}")
        return 1e6  # large penalty for failure


# -----------------------------
# 5. Define SSA search space
# -----------------------------
ngboost_x_min = [0.1, 10, 0.1]   # lower bounds: learning_rate, n_estimators, minibatch_frac
ngboost_x_max = [0.3, 300, 0.5]  # upper bounds

# Instantiate SSA optimizer
ssa_ngboost = SSA(
    N=10,            # number of sparrows (population size)
    dim=3,           # number of hyperparameters
    x_min=ngboost_x_min,
    x_max=ngboost_x_max,
    iterate_max=100  # max iterations
)

# -----------------------------
# 6. Run SSA optimization
# -----------------------------
print("Optimizing NGBoost...")
best_score_ngboost, best_params_ngboost = ssa_ngboost.optimize(
    fitness_function=ngboost_fitness
)

# -----------------------------
# 7. Print optimization results
# -----------------------------
print("Best hyperparameters for NGBoost:")
print(f"  Learning Rate: {best_params_ngboost[0]}")
print(f"  Number of Estimators: {int(best_params_ngboost[1])}")
print(f"  Minibatch Fraction: {best_params_ngboost[2]}")

# -----------------------------
# 8. Train final NGBoost model with best params
# -----------------------------
ngb = NGBRegressor(
    Dist=Normal,
    learning_rate=best_params_ngboost[0],
    n_estimators=int(best_params_ngboost[1]),
    minibatch_frac=best_params_ngboost[2]
)

# Train on full training set
ngb.fit(X_train_full, y_train_full)

# Evaluate on test set
y_pred_test = ngb.predict(X_test)
test_mse_ngboost = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE for NGBoost: {test_mse_ngboost}")

# -----------------------------
# 9. Save optimization results and model
# -----------------------------
sio.savemat('model_optimization_results_normal.mat', {
    'best_score_ngboost': best_score_ngboost,
    'best_params_ngboost': best_params_ngboost,
    'test_mse_ngboost': test_mse_ngboost
})
