# Importação das bibliotecas necessárias
import sys  # Biblioteca para manipulação do sistema
import os  # Biblioteca para manipulação de diretórios e arquivos
import numpy as np  # Biblioteca para manipulação de arrays
import pandas as pd  # Biblioteca para manipulação de dados
import joblib  # Biblioteca para salvar e carregar modelos e objetos

# Adiciona o diretório raiz ao sys.path para facilitar a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Carregamento do modelo e dos artefatos necessários para a inferência
model = joblib.load("artifacts/optimized_xgboost_model.pkl")

# Obtém as features esperadas pelo modelo e pelo imputador
expected_features = list(model.feature_names_in_)


# Função para realizar a inferência do modelo
def app_inferencia(input_data):
    input_df = pd.DataFrame(input_data, index=[0])

    input_df["Store"] = input_df["Store"].astype("int")
    input_df["Dept"] = input_df["Dept"].astype("int")
    input_df["Date"] = pd.to_datetime(input_df["Date"])
    input_df["IsHoliday"] = input_df["IsHoliday"].astype("bool")
    input_df["Temperature"] = input_df["Temperature"].astype("float")
    input_df["Fuel_Price"] = input_df["Fuel_Price"].astype("float")
    input_df["MarkDown1"] = np.nan
    input_df["MarkDown2"] = np.nan
    input_df["MarkDown3"] = np.nan
    input_df["MarkDown4"] = np.nan
    input_df["MarkDown5"] = np.nan
    input_df["CPI"] = input_df["CPI"].astype("float")
    input_df["Unemployment"] = input_df["Unemployment"].astype("float")
    input_df["Size"] = input_df["Size"].astype("int")

    # print("\n>>> DataFrame de entrada antes do pré-processamento:")
    # print(input_df)
    # Aplica a engenharia de atributos
    # input_df = seleciona_atributos(input_df)

    # Decomposição da variável data
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
    prediction = model.predict(input_df)[0]
    prediction = round(float(prediction), 2)
    print(f"\n>>> Previsão: {prediction:.2f}")

    return prediction
