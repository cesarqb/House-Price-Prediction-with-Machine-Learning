import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calcular_mape(y_real, y_pred):
    if any(y_real == 0):
        raise ValueError("Los valores reales no pueden contener ceros.")
    return np.mean(np.abs((y_real - y_pred) / y_real)) * 100


def evaluar_modelo(model, X_train, X_test, y_train, y_test):
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    resultados = {
        "MSE_train": mean_squared_error(y_train, y_pred_train),
        "MSE_test": mean_squared_error(y_test, y_pred_test),

        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),

        "MAPE_train": calcular_mape(y_train, y_pred_train),
        "MAPE_test": calcular_mape(y_test, y_pred_test),

        "R2_train": r2_score(y_train, y_pred_train),
        "R2_test": r2_score(y_test, y_pred_test)
    }

    return resultados, y_pred_train, y_pred_test