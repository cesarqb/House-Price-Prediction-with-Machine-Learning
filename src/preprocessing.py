# =============================
# FILE: src/preprocessing.py
# =============================

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import scipy.stats as stats
from scipy.stats.mstats import winsorize

def obtener_correlaciones(df, variables):
    """
    Genera un DataFrame con pares de variables y su correlación ordenada.
    """
    corr_matrix = df[variables].corr()
    corr_pairs = corr_matrix.unstack()
    
    # Eliminar duplicados y correlaciones consigo mismo
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    
    sorted_corr_pairs = corr_pairs.sort_values(ascending=False).drop_duplicates()
    
    result_df = pd.DataFrame(sorted_corr_pairs, columns=['Correlation'])
    result_df.reset_index(inplace=True)
    result_df.columns = ['Var1', 'Var2', 'Correlation']
    
    return result_df


def imputar_valores_extremos(df, variable, metodo='media'):
    if metodo not in ['media', 'mediana']:
        raise ValueError("El método debe ser 'media' o 'mediana'")

    if metodo == 'media':
        valor_imputacion = df[variable].mean()
    else:
        valor_imputacion = df[variable].median()

    limite_inferior = df[variable].mean() - 3 * df[variable].std()
    limite_superior = df[variable].mean() + 3 * df[variable].std()

    df[variable] = np.where(
        (df[variable] < limite_inferior) | (df[variable] > limite_superior),
        valor_imputacion,
        df[variable]
    )

    return df


def analizar_distribucion(df, variable):
    skew_val = skew(df[variable], nan_policy='omit')
    kurt_val = kurtosis(df[variable], nan_policy='omit')
    return skew_val, kurt_val



def feature_engineering(df):
    df = df.copy()

    # Transformaciones logarítmicas
    df["log_CRIM"] = np.log(df["CRIM"] + 1e-5)
    df["log_ZN"] = np.log(df["ZN"] + 1e-5)
    df["log_LSTAT"] = np.log(df["LSTAT"] + 1e-5)

    # Transformaciones raíz cuadrada
    df["sqrt_DIS"] = np.sqrt(df["DIS"])
    df["sqrt_INDUS"] = np.sqrt(df["INDUS"])

    # Box-Cox (requiere valores positivos)
    df['AGE_boxcox'], lam_age = stats.boxcox(df['AGE'])
    df['DIS_boxcox'], lam_dis = stats.boxcox(df['DIS'])
    df['LSTAT_boxcox'], lam_lstat = stats.boxcox(df['LSTAT'])

    # Lista de variables generadas
    var_engineering = [
        "log_CRIM", "log_ZN", "log_LSTAT",
        "sqrt_DIS", "sqrt_INDUS",
        "AGE_boxcox", "DIS_boxcox", "LSTAT_boxcox"
    ]

    lambdas = {
        "AGE": lam_age,
        "DIS": lam_dis,
        "LSTAT": lam_lstat
    }

    return df, var_engineering, lambdas



def preparar_nueva_data(df):

    # Copia
    df = df.copy()

    # Tipos
    df['CHAS'] = df['CHAS'].astype('object')

    # Eliminar columnas
    df = df.drop(['RAD', 'NOX'], axis=1)

    # Imputaciones
    df['ZN_imput'] = df['ZN']

    limite_inf = df['ZN_imput'].mean() - 3 * df['ZN_imput'].std()
    limite_sup = df['ZN_imput'].mean() + 3 * df['ZN_imput'].std()

    media = df['ZN_imput'].mean()

    df['ZN_imput'] = np.where(
        (df['ZN_imput'] < limite_inf) |
        (df['ZN_imput'] > limite_sup),
        media,
        df['ZN_imput']
    )

    # Winsorize
    df['ZN_imput_winsorized'] = winsorize(
        df['ZN'],
        limits=[0, 0.05]
    )

    # Log
    df["log_CRIM"] = np.log(df["CRIM"] + 1e-5)
    df["log_ZN"] = np.log(df["ZN"] + 1e-5)
    df["log_LSTAT"] = np.log(df["LSTAT"] + 1e-5)

    # Sqrt
    df["sqrt_DIS"] = np.sqrt(df["DIS"])
    df["sqrt_INDUS"] = np.sqrt(df["INDUS"])

    # Boxcox
    df['AGE_boxcox'], _ = stats.boxcox(df['AGE'])
    df['DIS_boxcox'], _ = stats.boxcox(df['DIS'])
    df['LSTAT_boxcox'], _ = stats.boxcox(df['LSTAT'])

    return df
