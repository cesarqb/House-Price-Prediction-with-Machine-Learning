# =============================
# FILE: src/outliers.py
# =============================

import numpy as np


def eliminar_filas_outliers(df, variable):
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = df[(df[variable] < limite_inferior) | (df[variable] > limite_superior)]
    porcentaje_outliers = len(outliers) / len(df) * 100

    if porcentaje_outliers > 5:
        return df[(df[variable] >= limite_inferior) & (df[variable] <= limite_superior)]
    return df.copy()


def porcentaje_outliers(df, variable):
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1

    v_col = df[(df[variable] <= q1 - 1.5 * iqr) | (df[variable] >= q3 + 1.5 * iqr)]
    return np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
