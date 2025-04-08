# Importação das bibliotecas necessárias
import numpy as np  # Biblioteca para operações numéricas
import pandas as pd  # Biblioteca para manipulação de dados em DataFrames
from sklearn.model_selection import (
    TimeSeriesSplit,
)  # Ferramenta para validação cruzada em séries temporais
from sklearn.metrics import mean_squared_error  # Métrica para avaliação do modelo


# Função para carregar os dados a partir de um arquivo CSV
def carrega_dados(file_path):
    return pd.read_csv(file_path)  # Retorna um DataFrame com os dados lidos


# Função para dividir os dados em conjuntos de treino e teste com base em uma data de corte
def time_series_split(data, cutoff_date):
    train = data.loc[data.index < cutoff_date]
    test = data.loc[data.index >= cutoff_date]
    return train, test


def time_series_cv(
    data,
    model,
    target,
    test_size=None,
    gap=0,
    n_splits=5,
    log=False,
    verbose=False,
    display_score=True,
):
    # Get sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    scores = []
    for fold, (train_index, val_index) in enumerate(tscv.split(data)):
        # Obtain train and validation data at fold k.
        train = data.iloc[train_index]
        val = data.iloc[val_index]

        # Obtain predictor and target train and validation sets.
        X_train = train.drop(columns=[target])
        y_train = train[target].copy()
        X_val = val.drop(columns=[target])
        y_val = val[target].copy()

        # Fit the model to the training data.
        model.fit(X_train, y_train)

        # Predict on validation data.
        y_pred = model.predict(X_val)

        # Obtain the validation score at fold k.
        if log:
            score = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred)))
        else:
            score = np.sqrt(mean_squared_error(y_val, y_pred))

        scores.append(score)

        # Print the results and returning scores array.

        if verbose:
            print("-" * 40)
            print(f"Fold {fold}")
            print(f"Score (RMSE) = {round(score, 4)}")

    if not display_score:
        return scores

    print("-" * 60)
    print(f"{type(model).__name__}'s time series cross validation results:")
    print(f"RMSE Média nos dados de validação = {round(np.mean(scores), 4)}")
    print(f"Desvio Padrão = {round(np.std(scores), 4)}")

    return scores
