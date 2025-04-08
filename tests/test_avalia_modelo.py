# Imports
import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from pipeline.avalia_modelo import avalia_modelo
from pipeline.preprocessa_dados import carrega_dados, time_series_split
from pipeline.engenharia_atributos import seleciona_atributos

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


def test_evaluate_model():
    model = joblib.load("artifacts/optimized_xgboost_model.pkl")

    # Carrega os dados a partir do caminho especificado na configuração
    train = carrega_dados(config["data"]["raw_path"]["train"])
    stores = carrega_dados(config["data"]["raw_path"]["stores"])
    features = carrega_dados(config["data"]["raw_path"]["features"])

    # Realiza o merge dos dados
    df_raw = pd.merge(train, features, how="inner", on=["Date", "Store"]).merge(
        stores, how="inner", on="Store"
    )

    # Realiza o pré-processamento dos dados
    df, _ = seleciona_atributos(df_raw)

    # Train-Test Split
    CUTOFF_DATE = config["training"]["cutoff_date"]
    train, test = time_series_split(df, CUTOFF_DATE)

    X_train = train.drop(columns=["Weekly_Sales"])
    y_train = train["Weekly_Sales"]
    X_test = test.drop(columns=["Weekly_Sales"])
    y_test = test["Weekly_Sales"]

    model.fit(X_train, y_train)

    # Faz as previsões
    y_pred = model.predict(X_test)

    metricas = avalia_modelo(y_test, y_pred)
    print(metricas)

    r2_score = metricas["R2 Score"]

    print(r2_score)

    assert r2_score > config["evaluation"]["r2_threshold"]


if __name__ == "__main__":
    test_evaluate_model()
