# Importação da biblioteca necessária
import pandas as pd  # Biblioteca para manipulação de dados em DataFrames


# Função para criação de novos atributos e remoção de colunas desnecessárias
# Veja a definição no Capítulo 10 do curso
def seleciona_atributos(df):
    # Remove a coluna
    cols_to_drop = [
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "IsHoliday_x",
        "IsHoliday_y",
    ]
    df.drop(cols_to_drop, axis=1, inplace=True)

    # Trabalhando apenas com valores maiores que 0
    df = df[(df["Weekly_Sales"] > 0) & (df["Weekly_Sales"] != 0)]

    # Vou remover os valores acima do percentil 98.
    df = df[df["Weekly_Sales"] < df["Weekly_Sales"].quantile(q=0.98)]

    # Decomposição da variável data
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter

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
        df[holiday] = df["Date"].isin(dates).astype(int)

    # Feature Engineering
    df = pd.get_dummies(df, columns=["Type"], dtype="int")
    df.set_index("Date", inplace=True)
    df.sort_values(by=["Date", "Store", "Dept"], inplace=True)

    # Retorna o DataFrame atualizado e a lista de colunas removidas
    return df, cols_to_drop
