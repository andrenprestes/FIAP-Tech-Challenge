import requests
import pandas as pd
import logging
import boto3
from datetime import datetime
from io import BytesIO

# Configuração do logging
class S3Handler(logging.Handler):
    """
    Um manipulador de log personalizado que envia mensagens de log para um bucket S3 da AWS.

    Atributos:
        bucket_name (str): O nome do bucket S3 onde os logs serão armazenados.
        object_name (str): O caminho e o nome do objeto no S3 para armazenar os logs.
        s3_client (boto3.client): Cliente S3 da biblioteca boto3.
        buffer (list): Lista temporária que armazena entradas de log até que sejam enviadas para o S3.
    """

    def __init__(self, bucket_name, object_name):
        """
        Inicializa um novo S3Handler.

        Parâmetros:
            bucket_name (str): O nome do bucket S3 onde os logs serão enviados.
            object_name (str): O nome do objeto no S3 para armazenar os logs.
        """
        super().__init__()
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.s3_client = boto3.client('s3')
        self.buffer = []

    def emit(self, record):
        """
        Emite uma entrada de log para o S3.

        Parâmetros:
            record (logging.LogRecord): O registro de log a ser emitido.
        """
        log_entry = self.format(record)
        self.buffer.append(log_entry)

        # Envia os logs para o S3 a cada 10 entradas ou ao final do programa
        if len(self.buffer) >= 10:
            self.flush()

    def flush(self):
        """
        Envia as entradas de log acumuladas para o S3 e limpa o buffer.
        """
        if self.buffer:
            log_content = '\n'.join(self.buffer)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=self.object_name, Body=log_content.encode('utf-8'))
            self.buffer.clear()


def get_data_from_api(api_url):
    """
    Faz uma requisição GET à API Flask e retorna os dados no formato JSON.

    Parâmetros:
        api_url (str): A URL da API onde os dados serão requisitados.

    Retorna:
        dict: Os dados JSON retornados pela API, se a requisição for bem-sucedida.
        None: Retorna None se ocorrer um erro na requisição ou se o código de status HTTP não for 200.
    """
    try:
        response = requests.get(api_url)

        # Verifica se a requisição foi bem-sucedida
        if response.status_code == 200:
            logging.info(f"Dados recebidos da API: {api_url}")
            return response.json()
        else:
            #print(f"Erro: Status Code {response.status_code} for {api_url}")
            logging.error(f"Erro: Status Code {response.status_code} ao acessar {api_url}")
            return None
    except Exception as e:
        #print(f"Erro ao acessar a API: {e}")
        logging.exception(f"Erro ao acessar a API: {e}")
        return None


def save_to_s3(df, bucket_name, file_key):
    """
    Salva um DataFrame como arquivo Parquet em um bucket S3.

    Parâmetros:
        df (pandas.DataFrame): O DataFrame a ser salvo no formato Parquet.
        bucket_name (str): O nome do bucket S3 onde o arquivo será salvo.
        file_key (str): O caminho e o nome do arquivo a ser salvo no bucket S3.
    """
    # Converte o DataFrame para formato Parquet
    buffer = BytesIO()
    # Convertendo todas as colunas para string
    df = df.astype(str)
    # Conversão em Parquet
    df.to_parquet(buffer, index=False)

    # Cria o cliente S3 usando boto3
    s3 = boto3.client('s3')

    try:
        # Faz o upload do arquivo Parquet no bucket S3
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=buffer.getvalue())
        logging.info(f"Arquivo {file_key} salvo com sucesso no bucket {bucket_name}.")
        #print(f"Arquivo {file_key} salvo com sucesso no bucket {bucket_name}.")
    except Exception as e:
        logging.exception(f"Erro ao salvar arquivo no S3: {e}")
        #print(f"Erro ao salvar arquivo no S3: {e}")


def main():
    """
    Função principal que faz a requisição à API Flask, processa os dados
    e os salva no AWS S3 como um arquivo Parquet.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Data do processamento
    dt_processamento = datetime.now().strftime('%Y-%m-%d')

    # Nome do bucket S3 e caminho do arquivo dos logs
    bucket_name = 'projeto-fiap-api'
    log_file_key = f'logs/{dt_processamento}/app.log'

    # Instânciando a Classe do Handler
    s3_handler = S3Handler(bucket_name, log_file_key)
    s3_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(s3_handler)

    lista_request = [
        'processamentoCSV/tipo/Viniferas',
        'processamentoCSV/tipo/Americanas',
        'processamentoCSV/tipo/Mesa',
        'processamentoCSV/tipo/Semclass',
        'producaoCSV',
        'comercializacaoCSV',
        'importacaoCSV/tipo/Vinhos',
        'importacaoCSV/tipo/Espumantes',
        'importacaoCSV/tipo/Frescas',
        'importacaoCSV/tipo/Passas',
        'importacaoCSV/tipo/Suco',
        'exportacaoCSV/tipo/Vinho',
        'exportacaoCSV/tipo/Espumantes',
        'exportacaoCSV/tipo/Uva',
        'exportacaoCSV/tipo/Suco'
    ]

    # URL da API Flask
    for item in lista_request:
        # TODO: Verifique o endereço real onde a API Flask está rodando.
        api_url = f'http://localhost:5000/{item}'

        # Caminho do Arquivo parquet
        file_key = f'data/{dt_processamento}/{item}.parquet'

        # Obtém os dados da API
        data = get_data_from_api(api_url)
        print(data)

        if data:
            if isinstance(data, dict):
                # Converte os dados JSON para um DataFrame do Pandas
                for key, value in data.items():
                    df = pd.DataFrame(value)
                    print(df.head())
                    df['data_processamento'] = dt_processamento

                    # Salva o DataFrame como Parquet no S3
                    save_to_s3(df, bucket_name, file_key)
            #elif isinstance(data, list):
            #    df = pd.DataFrame(data)
            #    print(df.head())
            #    df['data_processamento'] = dt_processamento

                # Salva o DataFrame como Parquet no S3
            #   save_to_s3(df, bucket_name, file_key)
            else:
                #print(f"Dataframe Vazio para {api_url}")
                logging.warning("Formato de dados inesperado. Esperado dict.")

    # Enviar logs restantes para o S3 antes de encerrar
    s3_handler.flush()

if __name__ == '__main__':
    main()
