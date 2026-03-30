# =============================
# FILE: src/preprocessing.py
# =============================

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


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