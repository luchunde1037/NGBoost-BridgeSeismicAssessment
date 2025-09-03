import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from ssa import SSA
import pandas as pd
import numpy as np
import random
# 定义 NGBoost 的适应度函数
from sklearn.exceptions import NotFittedError
import warnings
random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. Load preprocessed data
# -----------------------------
data = sio.loadmat("../../data/data_EDP1-1.mat")
X = data["X_scaled"]   # standardized features
y = data["y"].ravel()         # flatten target to 1D

# 3. 数据拆分为训练集和测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 进一步拆分训练集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)




def ngboost_fitness(params):
    learning_rate, n_estimators, minibatch_frac = params

    # 防止不合理参数（特别是小于 1 的 estimator 或过小的 batch）
    if n_estimators < 1 or minibatch_frac < 0.01:
        return 1e6  # 返回一个很差的分数，跳过这组参数

    try:
        # 初始化 NGBoost 模型
        ngb = NGBRegressor(
            Dist=LogNormal,
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            minibatch_frac=minibatch_frac,
            random_state=42,
            verbose=False
        )

        # 抑制 sklearn 的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ngb.fit(X_train, y_train)

        # 在验证集上进行预测并计算 MSE
        y_pred = ngb.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)

        # 如果损失是 NaN，也返回一个大值避免用它
        if np.isnan(mse):
            return 1e6
        return mse

    except (ValueError, NotFittedError, Exception) as e:
        print(f"[WARN] Fitness evaluation failed for params: {params}, error: {e}")
        return 1e6  # 返回极大值表示失败


# 定义 NGBoost 的搜索空间
ngboost_x_min = [0.1, 10, 0.1]
ngboost_x_max = [0.3, 300, 0.5]

# 实例化 SSA 类（用来进行超参数优化）
ssa_ngboost = SSA(N=10, dim=3, x_min=ngboost_x_min, x_max=ngboost_x_max, iterate_max=100)

# 使用 SSA 进行 NGBoost 的超参数优化
print("Optimizing NGBoost...")
best_score_ngboost, best_params_ngboost = ssa_ngboost.optimize(fitness_function=ngboost_fitness)

# 打印优化结果
print("Best hyperparameters for NGBoost:")
print(f"Learning Rate: {best_params_ngboost[0]}")
print(f"Number of Estimators: {int(best_params_ngboost[1])}")
print(f"Minibatch Fraction: {best_params_ngboost[2]}")

# 使用最佳超参数训练最终模型
ngb = NGBRegressor(
    Dist=LogNormal,
    learning_rate=best_params_ngboost[0],
    n_estimators=int(best_params_ngboost[1]),
    minibatch_frac=best_params_ngboost[2]
)

# 在整个训练集上训练 NGBoost 模型
ngb.fit(X_train_full, y_train_full)

# 在测试集上评估模型
y_pred_test = ngb.predict(X_test)
test_mse_ngboost = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE for NGBoost: {test_mse_ngboost}")

# 保存优化结果和模型
sio.savemat('model_optimization_results_lognormal.mat', {
    'best_score_ngboost': best_score_ngboost,
    'best_params_ngboost': best_params_ngboost,
    'test_mse_ngboost': test_mse_ngboost
})
