# Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import numpy as np
import pandas as pd
from pipeline.engenharia_atributos import seleciona_atributos


# pytest.fixture é um decorador do pytest, um framework de testes para Python.
# Ele é usado para criar fixtures, que são funções que preparam e fornecem dados ou objetos necessários para os testes.
@pytest.fixture
def sample_data():
    data = {
        "Store": [1, 2, 3],
        "Dept": [1, 4, 6],
        "Date": ["2010-02-05", "2010-02-12", "2010-02-19"],
        "Weekly_Sales": [24924.50, 46039.49, 41595.55],
        "IsHoliday_x": [False, True, False],
        "Temperature": [42.31, 38.51, 39.93],
        "Fuel_Price": [2.572, 2.548, 2.514],
        "MarkDown1": [np.nan, np.nan, np.nan],
        "MarkDown2": [np.nan, np.nan, np.nan],
        "MarkDown3": [np.nan, np.nan, np.nan],
        "MarkDown4": [np.nan, np.nan, np.nan],
        "MarkDown5": [np.nan, np.nan, np.nan],
        "CPI": [211.096358, 211.242170, 211.289143],
        "Unemployment": [8.106, 8.106, 8.106],
        "IsHoliday_y": [False, True, False],
        "Type": ["A", "B", "C"],
        "Size": [151315, 151315, 151315],
    }
    return pd.DataFrame(data)


def test_select_features(sample_data):
    df_selected, dropped_features = seleciona_atributos(sample_data)

    assert "Temperature" in dropped_features
    assert "MarkDown1" in dropped_features
    assert "Unemployment" in dropped_features

    assert "Store" in df_selected.columns
    assert "SuperBowl" in df_selected.columns
    assert "Size" in df_selected.columns
