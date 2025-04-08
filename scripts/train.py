# Importação das bibliotecas necessárias
import os
import sys
import yaml
import logging
import logging.config
import pandas as pd
import xgboost as xgb

# Adiciona o diretório raiz ao sys.path para facilitar a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importação das funções do pipeline
from pipeline.preprocessa_dados import (
    carrega_dados,
    time_series_split,
    time_series_cv,
)
from pipeline.engenharia_atributos import seleciona_atributos
from pipeline.avalia_modelo import avalia_modelo
from pipeline.otimiza_hiperparametros import hyperparameter_tuning
from pipeline.salva_artefatos import salva_artefatos


# Carrega a configuração de logging a partir do arquivo YAML
with open("config/logging.yaml", "r") as file:
    logging.config.dictConfig(yaml.safe_load(file))

logger = logging.getLogger(__name__)

# Carrega a configuração do pipeline a partir do arquivo YAML
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Função principal do pipeline
def main():
    try:
        logger.info("Carregando e Pré-processando Dados...")

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

        # Treinamento Teste e Validação
        scores = time_series_cv(
            train,
            xgb.XGBRegressor(n_jobs=-1, verbosity=0),
            config["training"]["target"],
            config["training"]["test_size"],
            n_splits=5,
        )

        X_train = train.drop(columns=["Weekly_Sales"])
        y_train = train["Weekly_Sales"]
        X_test = test.drop(columns=["Weekly_Sales"])
        y_test = test["Weekly_Sales"]

        logger.info("Ajuste de Hiperparâmetros Para o Melhor Modelo...")
        best_params = hyperparameter_tuning(train)

        optimized_model = xgb.XGBRegressor(**best_params, n_jobs=-1)
        optimized_model.fit(X_train, y_train)

        # Faz as previsões
        y_pred = optimized_model.predict(X_test)

        # Avalia o modelo
        metricas = avalia_modelo(y_test, y_pred)

        # Log
        logger.info(f"Métricas de Avaliação do Modelo: {metricas}")

        # Define os caminhos para salvar o modelo e os artefatos
        model_path = config["model"]["path"]

        # Salva os artefatos do modelo
        salva_artefatos(optimized_model, model_path)

        # Salva os dados de treino e teste em arquivos CSV
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(
            config["data"]["processed_path"] + "/train_data.csv", index=False
        )
        test_data.to_csv(
            config["data"]["processed_path"] + "/test_data.csv", index=False
        )

        logger.info("Treinamento Concluído com Sucesso!")

    except Exception as e:
        logger.exception("Ocorreu uma exceção durante o treinamento: %s", str(e))


# Executa a função principal quando o script for rodado diretamente
if __name__ == "__main__":
    main()
