
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
from datetime import datetime


## def LinearRegression
## --------------------
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def get_model_coefficients(model, feature_names):
    import pandas as pd
    
    coef_df = pd.DataFrame({
        "Variable": feature_names,
        "Coeficiente": model.coef_
    })
    
    intercepto = model.intercept_
    
    return intercepto, coef_df


## def Rige regression
## -------------------

def generar_alphas():
    return 10**np.linspace(10, -1, 100) * 0.5


def ridge_path(X, y, alphas):
    """
    Calcula los coeficientes para distintos valores de alpha
    (para graficar regularización)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge()
    coefs = []

    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X_scaled, y)
        coefs.append(ridge.coef_)

    return np.array(coefs), alphas


def train_ridge_model(X_train, y_train, alphas):
    """
    Entrena Ridge con selección automática de alpha (CV)
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
    ridgecv.fit(X_train_scaled, y_train)

    best_alpha = ridgecv.alpha_

    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)

    return model, scaler, best_alpha


def predict_ridge(model, scaler, X_train, X_test):
    """
    Genera predicciones usando el mismo scaler
    """
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    return y_pred_train, y_pred_test



## def Lasso regression
## --------------------

def lasso_path(X_train_scaled, y_train, alphas):
    """
    Calcula coeficientes para distintos alpha (para gráfico)
    """
    coefs = []
    lasso = Lasso(max_iter=10000)

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train_scaled, y_train)
        coefs.append(lasso.coef_)

    return np.array(coefs)


def train_lasso_model(X_train, y_train):
    """
    Entrena Lasso con selección automática de alpha (CV)
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lassocv = LassoCV(cv=10, max_iter=100000)
    lassocv.fit(X_train_scaled, y_train)

    best_alpha = lassocv.alpha_

    model = Lasso(alpha=best_alpha, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, best_alpha


def predict_lasso(model, scaler, X_train, X_test):
    """
    Predicciones usando mismo scaler
    """
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    return y_pred_train, y_pred_test



def guardar_modelo(objeto, nombre_modelo="modelo"):
    """
    Guarda el modelo en la carpeta /models en la raíz del proyecto
    """

    # Obtener ruta base del proyecto (sube desde /notebooks o /src)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Crear ruta correcta
    carpeta = os.path.join(base_path, "models")
    os.makedirs(carpeta, exist_ok=True)

    # Nombre con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{nombre_modelo}_{timestamp}.pkl"

    ruta = os.path.join(carpeta, filename)

    # Guardar
    with open(ruta, "wb") as f:
        pickle.dump(objeto, f)

    print(f" Modelo guardado en: {ruta}")

    return ruta