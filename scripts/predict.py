# Importação das bibliotecas necessárias
import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# Adiciona o diretório raiz ao sys.path para facilitar a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Carregamento do modelo e dos artefatos necessários para a inferência
model = joblib.load(
    "artifacts/optimized_xgboost_model.pkl"
)  # Carrega o modelo treinado

# Obtém as features esperadas pelo modelo e pelo imputador
expected_features = list(model.feature_names_in_)


# Função para realizar a inferência do modelo
def inferencia(input_data):
    input_df = pd.DataFrame(input_data, index=[0])

    # print("\n>>> DataFrame de entrada antes do pré-processamento:")
    # print(input_df)
    # Aplica a engenharia de atributos
    # input_df = seleciona_atributos(input_df)

    # Decomposição da variável data
    input_df["Date"] = pd.to_datetime(input_df["Date"])
    input_df["Week"] = input_df["Date"].dt.isocalendar().week
    input_df["Month"] = input_df["Date"].dt.month
    input_df["Quarter"] = input_df["Date"].dt.quarter

    # Criando colunas de feriados
    holiday_dates = {
        "SuperBowl": ["2010-02-12", "2011-02-11", "2012-02-10"],
        "LaborDay": ["2010-09-10", "2011-09-09", "2012-09-07"],
        "Thanksgiving": ["2010-11-26", "2011-11-25"],
        "Christmas": ["2010-12-31", "2011-12-30"],
    }

    # Convertendo as datas para datetime
    holiday_dates = {
        key: pd.to_datetime(values) for key, values in holiday_dates.items()
    }

    # Criando as colunas de feriados usando .isin()
    for holiday, dates in holiday_dates.items():
        input_df[holiday] = input_df["Date"].isin(dates).astype(int)

    # Feature Engineering
    input_df = pd.get_dummies(input_df, columns=["Type"], dtype="int")
    input_df.set_index("Date", inplace=True)
    input_df.sort_values(by=["Date", "Store", "Dept"], inplace=True)

    # Remove a coluna
    cols_to_drop = [
        "IsHoliday",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
    ]
    input_df.drop(cols_to_drop, axis=1, inplace=True)

    # Reorganiza as colunas para corresponder às esperadas pelo modelo
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # print("\n>>> DataFrame final enviado ao modelo:")
    # print(input_df.head())

    # Faz a previsão usando o modelo treinado
    prediction = model.predict(input_df)
    prediction = round(prediction[0], 2)

    return prediction


# Função principal para execução via linha de comando
def main():
    input_data = {
        "Store": 1,
        "Dept": 1,
        "Date": "2010-02-19",
        "IsHoliday": False,
        "Temperature": 39.93,
        "Fuel_Price": 2.514,
        "MarkDown1": np.nan,
        "MarkDown2": np.nan,
        "MarkDown3": np.nan,
        "MarkDown4": np.nan,
        "MarkDown5": np.nan,
        "CPI": 211.289143,
        "Unemployment": 8.106,
        "Type": "A",
        "Size": 151315,
    }

    prediction = inferencia(input_data)
    print("\n>>> Previsão do modelo:", prediction)
    print("\n>>> Margem de erro p/ Baixo:", prediction - 2000)
    print("\n>>> Margem de erro p/ Cima:", prediction + 2000)


# Executa a função principal quando o script for rodado diretamente
if __name__ == "__main__":
    main()
