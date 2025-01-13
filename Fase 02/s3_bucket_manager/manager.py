"""
Módulo para operações com o AWS S3 e manipulação de arquivos Parquet.

Este módulo fornece funções utilitárias para interagir com buckets do AWS S3,
realizar operações como upload, download, exclusão e listagem de arquivos,
bem como salvar um DataFrame em formato Parquet com partições diárias.

Dependências:
    - boto3: Biblioteca para interagir com serviços da AWS.
    - botocore: Utilizada para lidar com exceções específicas do cliente AWS.
    - pandas: Para manipulação de DataFrames.
    - pyarrow: Para conversão de DataFrames em arquivos Parquet.
    - os: Para manipulação de caminhos de arquivos e diretórios.

Funções:
    - get_s3_client(): Retorna um cliente S3 usando boto3.
    - get_region(): Obtém a região padrão da sessão atual do AWS.
    - ensure_bucket_exists(bucket_name): Verifica se um bucket existe, criando-o se necessário.
    - create_bucket(bucket_name): Cria um bucket no S3.
    - check_bucket_exists(bucket_name): Verifica se um bucket existe no S3.
    - upload_file(bucket_name, file_path, s3_key): Faz upload de um arquivo local para o S3.
    - download_file(bucket_name, s3_key, file_path): Baixa um arquivo do S3 para o local.
    - list_files(bucket_name, prefix=""): Lista arquivos em um bucket com um prefixo opcional.
    - delete_file(bucket_name, s3_key): Exclui um arquivo específico de um bucket.
    - delete_folder(bucket_name, folder_name): Exclui todos os arquivos em uma pasta no bucket.
    - file_exists(bucket_name, s3_key): Verifica se um arquivo específico existe no bucket.
    - delete_bucket(bucket_name): Exclui um bucket S3 vazio.
    - empty_bucket(bucket_name): Remove todos os arquivos de um bucket.
    - save_dataframe_to_parquet(df, bucket_name, s3_path): Salva um DataFrame como Parquet no S3, criando partições diárias.

Observações:
    - As funções lidam com exceções comuns como credenciais ausentes (NoCredentialsError) e erros do cliente (ClientError).
    - Para salvar um DataFrame como Parquet, a função `save_dataframe_to_parquet` manipula colunas específicas e formatações,
      garantindo compatibilidade com o formato Parquet e as partições S3.

Notas importantes:
    - Certifique-se de configurar corretamente suas credenciais AWS antes de usar este módulo.
    - Garanta que os nomes de buckets e chaves estejam corretos e sejam consistentes com seu ambiente.
"""
from botocore.exceptions import NoCredentialsError, ClientError
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import boto3
import os

current_folder_path = os.getcwd()
download_dir = os.path.join(current_folder_path, "IBOVDia")

def get_s3_client():
    """
    Returns an S3 client using the boto3 library.

    Returns:
        boto3.client: Configured S3 client for the current AWS session.
    """
    return boto3.client('s3')

def get_region():
    """
    Retrieves the default region configured in the current AWS session.

    Returns:
        str: Default AWS region for the current session.
    """
    return boto3.session.Session().region_name

def ensure_bucket_exists(bucket_name):
    """
    Checks if an S3 bucket exists and creates it if it does not.

    Args:
        bucket_name (str): Name of the S3 bucket to check/create.
    """
    if not check_bucket_exists(bucket_name):
        print(f"Bucket {bucket_name} não existe. Criando...")
        create_bucket(bucket_name)

def create_bucket(bucket_name):
    """
    Creates an S3 bucket.

    Args:
        bucket_name (str): Name of the bucket to create.

    Raises:
        NoCredentialsError: If AWS credentials are not found.
        Exception: For other errors during bucket creation.
    """
    try:
        s3_client = get_s3_client()
        s3_client.create_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} criado com sucesso!')
    except NoCredentialsError:
        print('Credenciais não encontradas. Verifique suas chaves de acesso.')
    except Exception as e:
        print(f'Erro ao criar o bucket: {e}')

def check_bucket_exists(bucket_name):
    """
    Checks if an S3 bucket exists.

    Args:
        bucket_name (str): Name of the S3 bucket to check.

    Returns:
        bool: True if the bucket exists, False otherwise.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} existe.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Bucket {bucket_name} não encontrado.")
            return False
        else:
            print(f"Erro ao verificar bucket: {e}")
            return False

def upload_file(bucket_name, file_path, s3_key):
    """
    Uploads a local file to an S3 bucket.

    Args:
        bucket_name (str): Name of the destination S3 bucket.
        file_path (str): Local path of the file to upload.
        s3_key (str): S3 key (path in the bucket) for the uploaded file.

    Raises:
        NoCredentialsError: If AWS credentials are not found.
        ClientError: If there is an error with the AWS client.
    """
    try:
        s3_client = get_s3_client()
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"Arquivo {file_path} enviado para {bucket_name}/{s3_key}")
    except NoCredentialsError:
        print("Credenciais AWS não encontradas.")
    except ClientError as e:
        print(f"Erro ao fazer upload: {e}")

def download_file(bucket_name, s3_key, file_path):
    """
    Downloads a file from an S3 bucket to the local system.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 key (path in the bucket) of the file to download.
        file_path (str): Local path where the file will be saved.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        s3_client = get_s3_client()
        s3_client.download_file(bucket_name, s3_key, file_path)
        print(f"Arquivo {s3_key} baixado para {file_path}")
    except ClientError as e:
        print(f"Erro ao fazer download: {e}")

def list_files(bucket_name, prefix=""):
    """
    Lists files in an S3 bucket, optionally filtered by a prefix.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str, optional): Prefix to filter files. Defaults to "".

    Returns:
        list[str]: List of file keys found in the bucket.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        s3_client = get_s3_client()
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            print(f"Arquivos encontrados no bucket {bucket_name}:")
            for file in files:
                print(file)
            return files
        else:
            print("Nenhum arquivo encontrado.")
            return []
    except ClientError as e:
        print(f"Erro ao listar arquivos: {e}")
        return []

def delete_file(bucket_name, s3_key):
    """
    Deletes a specific file from an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 key (path in the bucket) of the file to delete.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        s3_client = get_s3_client()
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print(f"Arquivo {s3_key} excluído do bucket {bucket_name}")
    except ClientError as e:
        print(f"Erro ao excluir arquivo: {e}")

def delete_folder(bucket_name, folder_name):
    """
    Deletes all files within a folder in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        folder_name (str): Name of the folder (prefix) to delete.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        files = list_files(bucket_name, prefix=folder_name)
        for file in files:
            delete_file(bucket_name, file)
        print(f"Pasta {folder_name} excluída do bucket {bucket_name}")
    except ClientError as e:
        print(f"Erro ao excluir pasta: {e}")

def file_exists(bucket_name, s3_key):
    """
    Checks if a specific file exists in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 key (path in the bucket) of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"Arquivo {s3_key} existe no bucket {bucket_name}.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Arquivo {s3_key} não encontrado no bucket {bucket_name}.")
            return False
        else:
            print(f"Erro ao verificar arquivo: {e}")
            return False

def delete_bucket(bucket_name):
    """
    Deletes an empty S3 bucket.

    Args:
        bucket_name (str): Name of the bucket to delete.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        s3_client = get_s3_client()
        s3_client.delete_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} excluído com sucesso!")
    except ClientError as e:
        print(f"Erro ao excluir o bucket: {e}")

def empty_bucket(bucket_name):
    """
    Removes all files from an S3 bucket.

    Args:
        bucket_name (str): Name of the bucket to clear.

    Raises:
        ClientError: If there is an error with the AWS client.
    """
    try:
        files = list_files(bucket_name)
        for file in files:
            delete_file(bucket_name, file)
        print(f"Bucket {bucket_name} limpo com sucesso!")
    except ClientError as e:
        print(f"Erro ao limpar o bucket: {e}")

def save_dataframe_to_parquet(df, bucket_name, s3_path):
    """
    Saves a DataFrame as a Parquet file to S3 with daily partitioning.

    Args:
        df (pandas.DataFrame): DataFrame to save.
        bucket_name (str): Name of the S3 bucket.
        s3_path (str): S3 path where the file will be saved.

    Raises:
        Exception: For general errors during processing or upload.
    """
    try:
        df['Date'] = pd.Timestamp.today().strftime('%Y-%m-%d')
        print(df.columns)
        df.columns = ['Codigo', 'Acao', 'Tipo', 'Qtde_Teorica', 'Part', 'Date']
        df['Qtde_Teorica'] = df['Qtde_Teorica'].str.replace('.', '', regex=False).astype(float)
        df['Part'] = df['Part'].str.replace(',', '.', regex=False).astype(float)
        table = pa.Table.from_pandas(df)
        date_str = pd.Timestamp.today().strftime('%Y-%m-%d')
        partitioned_path = f"{s3_path}/date={date_str}/data.parquet"
        parquet_file = 'data.parquet'
        
        # Usar caminho absoluto para o diretório temporário
        download_dir = os.path.join(os.getcwd(), 'temp')
        temp_file = os.path.join(download_dir, parquet_file)
        
        # Criar diretório se não existir
        os.makedirs(download_dir, exist_ok=True)
        
        # Salvar arquivo Parquet localmente
        pq.write_table(table, temp_file)

        s3 = get_s3_client()
        print(f"Iniciando upload para: {partitioned_path}")
        
        # Abrir arquivo em modo binário
        with open(temp_file, 'rb') as f:
            object_data = f.read()
            s3.put_object(Body=object_data, Bucket=bucket_name, Key=partitioned_path)
    
        print(f'Arquivo salvo com sucesso: s3://{bucket_name}/{partitioned_path}')
        
    except Exception as e:
        print(f"Erro ao fazer upload do arquivo para o S3: {str(e)}")
        raise
    finally:
        # Limpar arquivo temporário se ele existir
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Arquivo temporário removido: {temp_file}")
