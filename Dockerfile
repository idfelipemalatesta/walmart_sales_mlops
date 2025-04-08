# Use uma imagem oficial do Python como base
FROM python:3.12

# Definir o diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos para dentro do container
COPY requirements.txt requirements.txt
COPY . /app

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta para acesso externo
EXPOSE 5002

# Comando para rodar a API Flask
CMD ["python", "app/app.py"]
