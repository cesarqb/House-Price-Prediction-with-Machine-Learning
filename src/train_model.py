from sklearn.linear_model import LinearRegression

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