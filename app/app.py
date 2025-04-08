# Importação das bibliotecas necessárias
import sys
import os
from flask import (
    Flask,
    request,
    render_template,
)

# Adiciona o diretório raiz ao sys.path para facilitar a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importação da função de inferência
from scripts.app_inference import app_inferencia

import warnings

warnings.filterwarnings("ignore")


# Inicializa a aplicação Flask
app = Flask(__name__)


# Define a rota principal da aplicação
@app.route("/")
def home():
    return render_template("index.html")  # Renderiza a página inicial


# Define a rota para fazer previsões
@app.route("/predict", methods=["POST"])
def predict():
    # Obtém os dados enviados pelo formulário na requisição
    data = request.form.to_dict()

    # Exibe os dados recebidos no console para depuração
    print("Received data:", data)

    # Realiza a previsão com os dados processados
    prediction = app_inferencia(data)

    # Renderiza a página com o resultado da previsão
    return render_template(
        "index.html", prediction_text=f"Valor Previsto: ${prediction}"
    )


# Executa a aplicação Flask no modo debug se o script for executado diretamente
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
