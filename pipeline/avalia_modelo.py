# Importação das métricas para avaliação de modelos
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)  # Importa funções para cálculo de erro e desempenho


# Função para avaliar o desempenho do modelo
def avalia_modelo(y_test, y_pred):
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    rmse = round(root_mean_squared_error(y_test, y_pred), 4)
    mape = round(mean_absolute_percentage_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)

    # Retorna um dicionário com as métricas calculadas
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2 Score": r2}
