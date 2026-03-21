# PREDICCION DE PRECIOS DE VIVIENDA CON MACHINE LEARNING

Este proyecto implementa un flujo completo de análisis de datos y Machine Learning para predecir precios de viviendas a partir de variables estructurales.

El proyecto incluye desde el análisis exploratorio de datos (EDA), tratamiento de outliers, selección de variables y escalamiento, hasta la construcción y evaluación de modelos de regresión.

Además, el modelo es validado utilizando un segundo dataset (vivienda_new), permitiendo evaluar su capacidad de generalización en datos no vistos.

---

## Objetivos del proyecto

- Analizar datos de viviendas y su relación con el precio.
- Detectar y tratar valores atípicos (outliers).
- Aplicar análisis exploratorio de datos (EDA).
- Evaluar correlaciones entre variables.
- Construir modelos predictivos de regresión.
- Comparar distintos modelos (Ridge, Lasso, etc.).
- Validar el modelo con datos nuevos.

---

## Técnicas utilizadas

Durante el proyecto se aplican diferentes técnicas de Ciencia de Datos:

- Limpieza y transformación de datos
- Detección e imputación de outliers
- Análisis de correlación
- Visualización de datos (histogramas, heatmaps)
- Escalado de variables (StandardScaler)
- Modelos de regresión:
    - Regresión Lineal
    - Ridge Regression
    - Lasso Regression
    - Validación cruzada (Cross Validation)
    - Evaluación de modelos: MSE, RMSE, MAPE
- Análisis de residuos
- Serialización del modelo (pickle)

---

## Tecnologías utilizadas

- Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn, Pickle

---

## Estructura del proyecto

```
House-Price-Prediction-with-Machine-Learning
│
├── README.md
├── requirements.txt
├── housing_price_notebook.ipynb
│
├── data
│   ├── raw
│   │   ├── vivienda.csv
│   │   └── vivienda_new.csv
│   │
│   └── processed
│       └── vivienda_clean.csv
│
├── notebooks
│   └── housing_price_modeling.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── outliers.py
│   ├── train_model.py
│   └── evaluation.py
│
├── images
│
├── models
│   └── house_price_model.pkl
│
└── results
    └── predictions.csv
```

---

## Descripción de carpetas

```
data/ → contiene datos originales (vivienda, vivienda_new) y procesados.
notebooks/ → análisis exploratorio y desarrollo del modelo.
src/ → funciones reutilizables (preprocesamiento, modelado, evaluación).
images/ → visualizaciones generadas durante el análisis.
models/ → modelo entrenado guardado con pickle.
results/ → predicciones y métricas finales.
```
---

## Flujo del análisis

El análisis sigue las siguientes etapas:

1. Carga de datos (vivienda)
2. Limpieza de datos y transformación de variables
3. Detección y tratamiento de outliers
4. Análisis exploratorio (EDA)
5. Análisis de correlaciones
6. Selección de variables
7. Escalado de datos
8. División Train/Test
9. Entrenamiento de modelos: Ridge, Lasso
10. Optimización de hiperparámetros
11. Evaluación del modelo
12. Análisis de residuos
13. Validación con dataset nuevo (vivienda_new)
14. Guardado del modelo

---

## Ejemplo de resultados

Precio real: 22.5  
Precio predicho: 21.8  

Error: 0.7

---

## Posibles mejoras futuras

- Implementar modelos más avanzados (XGBoost, LightGBM)
- Automatizar pipeline con Scikit-learn Pipeline
- Feature selection más robusta
- Deploy del modelo (API con FastAPI o Streamlit)
- Integración con datos geográficos

---

## 👨‍🏫 Autor

César Quezada
Científico de Datos | Docente Universitario | Mentor
