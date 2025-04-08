# Walmart Sales - Automação das Operações de CI/CD no Pipeline de Machine Learning



## Estrutura do Projeto
```
│── .github/
│   └── workflows/
│       └── mlops-pipeline.yml    			# Arquivo do GitHub Actions para automação
│
├── artifacts/                    			# Armazena modelos treinados e objetos de pré-processamento
│   ├── optimized_lightgbm_model.pkl
│
├── config/                       			# Arquivos de configuração
│   ├── config.yaml               			# Hiperparâmetros de modelo e pré-processamento
│   ├── logging.yaml              			# Configurações de log
│
├── data/                         			# Dados brutos e processados
│   ├── raw/
│   │   ├── train.csv
|   |   ├── stores.csv
|   |   ├── features.csv
│   ├── processed/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│
├── pipeline/                          	# Código-fonte para pipeline de ML modularizado
│   ├── __init__.py
│   ├── preprocessa_dados.py     		# Manipulando valores ausentes, codificação, dimensionamento
│   ├── engenharia_atributos.py    		# Seleção de recursos, transformação
│   ├── otimiza_hiperparametros.py  	# Optuna para ajuste de hiperparâmetros
│   ├── avalia_modelo.py         		# Avaliação de modelo e análise residual
│   ├── salva_artefatos.py         		# Salvar modelo e artefatos de pré-processamento
│
├── scripts/                      		# Scripts de automação
│   ├── train.py        	            # Automatiza o pipeline de treinamento
│   ├── predict.py                      # Automatiza o pipeline de inferência (previsões)
│   ├── app_inference.py    			# Usado para previsões na app
│
├── app/                          			# Deploy do modelo via app web com Flask
│   ├── app.py                    			# Endpoint de inferência
│   ├── templates/
│       └── index.html            			# Página HTML para inputs do usuário
│
├── tests/                        			# Testes unitários para o Pipeline CI/CD
│   ├── test_engenharia_atributos.py
│   ├── test_avalia_modelo.py
│
├── Dockerfile                    			# Dockerfile para criação de container
├── requirements.txt              			# Dependências
├── LEIAME.txt                     			# Documentação do projeto
```



# Abra o terminal ou prompt de comando, navegue até a pasta com os arquivos e execute o comando abaixo para criar um ambiente virtual:

```
conda create --name walmart_sales python=3.12
```

# Ative o ambiente:

```
conda activate walmart_sales
```

# Instale o pip e as dependências:

```
pip install -r requirements.txt 
```

# Execute os os comandos abaixo:

python scripts/train.py
python scripts/predict.py
pytest -v
python app/app.py
docker build -t walmart_sales-mlops .
docker run -d -p 5002:5002 --name walmart-mlops walmart_sales-mlops
