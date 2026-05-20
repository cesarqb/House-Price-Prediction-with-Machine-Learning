import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def calcular_mape(y_real, y_pred):
    if any(y_real == 0):
        raise ValueError("Los valores reales no pueden contener ceros.")
    return np.mean(np.abs((y_real - y_pred) / y_real)) * 100


def evaluar_modelo(model, X_train, X_test, y_train, y_test, verbose=True):
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    mape_train = calcular_mape(y_train, y_pred_train)
    mape_test = calcular_mape(y_test, y_pred_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    resultados = {
        "MSE_train": mse_train,
        "MSE_test": mse_test,
        "RMSE_train": rmse_train,
        "RMSE_test": rmse_test,
        "MAPE_train": mape_train,
        "MAPE_test": mape_test,
        "R2_train": r2_train,
        "R2_test": r2_test
    }

    # Modo reporte bonito
    if verbose:
        print("\n EVALUACIÓN DEL MODELO\n")

        print("Error cuadrático medio (MSE):")
        print(f"Train: {mse_train:.4f}")
        print(f"Test:  {mse_test:.4f}\n")

        print("Raíz del error cuadrático medio (RMSE):")
        print(f"Train: {rmse_train:.4f}")
        print(f"Test:  {rmse_test:.4f}\n")

        print("Error porcentual absoluto medio (MAPE):")
        print(f"Train: {mape_train:.2f}%")
        print(f"Test:  {mape_test:.2f}%\n")

        print("Coeficiente de determinación (R²):")
        print(f"Train: {r2_train:.4f}")
        print(f"Test:  {r2_test:.4f}")

    return resultados, y_pred_train, y_pred_test


def analizar_residuos(y_true, y_pred, feature=None, feature_name=None):
    
    residuos = y_true - y_pred

    print("\n ANÁLISIS DE RESIDUOS\n")

    print(f"Media de residuos: {np.mean(residuos):.4f}")
    print(f"Desviación estándar: {np.std(residuos):.4f}")
    print(f"Asimetría (skewness): {stats.skew(residuos):.4f}")
    print(f"Curtosis: {stats.kurtosis(residuos):.4f}")

    # -----------------------------
    # 1. Predicción vs Real
    # -----------------------------
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_pred, y=y_true, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Predicción vs Valores Reales')
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')
    plt.show()

    # -----------------------------
    # 2. Residuos vs Predicción
    # -----------------------------
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_pred, y=residuos, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuos vs Predicción')
    plt.xlabel('Predicción')
    plt.ylabel('Residuos')
    plt.show()

    # -----------------------------
    # 3. Histograma
    # -----------------------------
    plt.figure(figsize=(8,6))
    sns.histplot(residuos, kde=True)
    plt.title("Distribución de residuos")
    plt.show()

    # -----------------------------
    # 4. Q-Q plot
    # -----------------------------
    plt.figure(figsize=(8,6))
    stats.probplot(residuos, dist="norm", plot=plt)
    plt.title("Q-Q plot")
    plt.show()

    # -----------------------------
    # 5. Residuos vs variable (opcional)
    # -----------------------------
    if feature is not None:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=feature, y=residuos, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuos vs {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Residuos')
        plt.show()

    return residuos


def detectar_viviendas_subvaloradas(X, y_real, y_pred):
    """
    Detecta viviendas cuyo precio real es menor al predicho
    """

    df = X.copy()

    df["PRICE_REAL"] = y_real
    df["PRICE_PRED"] = y_pred

    # Error
    df["ERROR"] = df["PRICE_PRED"] - df["PRICE_REAL"]

    # Porcentaje de subvaloración
    df["PORCENTAJE_SUBVALORACION"] = (
        df["ERROR"] / df["PRICE_REAL"]
    ) * 100

    # Filtrar subvaloradas
    subvaloradas = df[df["PRICE_REAL"] < df["PRICE_PRED"]]

    # Ordenar
    subvaloradas = subvaloradas.sort_values(
        by="PORCENTAJE_SUBVALORACION",
        ascending=False
    )

    return subvaloradas



def plot_subvaloradas(subvaloradas):
    
    plt.figure(figsize=(10,6))

    sns.histplot(
        subvaloradas['ERROR'],
        bins=30,
        kde=True
    )

    plt.title('Distribución de errores - viviendas subvaloradas')
    plt.xlabel('Error (Predicción - Real)')
    plt.ylabel('Frecuencia')

    plt.axvline(
        subvaloradas['ERROR'].mean(),
        color='red',
        linestyle='--',
        label=f"Media: {subvaloradas['ERROR'].mean():.2f}"
    )

    plt.legend()
    plt.show()