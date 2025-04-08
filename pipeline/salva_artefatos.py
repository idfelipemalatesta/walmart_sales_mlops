# Importação das bibliotecas necessárias
import joblib
import os


# Função para salvar o modelo e os artefatos de pré-processamento
def salva_artefatos(
    model,
    model_path,
):
    # Garante que o diretório de destino existe, criando-o se necessário
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Salva o modelo treinado no caminho especificado
    joblib.dump(model, model_path)

    # Exibe uma mensagem indicando que os artefatos foram salvos com sucesso
    print("Modelo Salvo com Sucesso!")
