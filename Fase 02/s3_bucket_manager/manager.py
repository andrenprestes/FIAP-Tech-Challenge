import boto3
from botocore.exceptions import NoCredentialsError, ClientError

import pandas as pd
import boto3
import os
import pyarrow as pa
import pyarrow.parquet as pq

current_folder_path = os.getcwd()
download_dir = os.path.join(current_folder_path, "IBOVDia")

def get_s3_client():
    return boto3.client('s3')

def get_region():
    return boto3.session.Session().region_name

def ensure_bucket_exists(bucket_name):
    if not check_bucket_exists(bucket_name):
        print(f"Bucket {bucket_name} não existe. Criando...")
        create_bucket(bucket_name)

def create_bucket(bucket_name):
    try:
        s3_client = get_s3_client()
        s3_client.create_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} criado com sucesso!')
    except NoCredentialsError:
        print('Credenciais não encontradas. Verifique suas chaves de acesso.')
    except Exception as e:
        print(f'Erro ao criar o bucket: {e}')

def check_bucket_exists(bucket_name):
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
    try:
        s3_client = get_s3_client()
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"Arquivo {file_path} enviado para {bucket_name}/{s3_key}")
    except NoCredentialsError:
        print("Credenciais AWS não encontradas.")
    except ClientError as e:
        print(f"Erro ao fazer upload: {e}")

def download_file(bucket_name, s3_key, file_path):
    try:
        s3_client = get_s3_client()
        s3_client.download_file(bucket_name, s3_key, file_path)
        print(f"Arquivo {s3_key} baixado para {file_path}")
    except ClientError as e:
        print(f"Erro ao fazer download: {e}")

def list_files(bucket_name, prefix=""):
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
    try:
        s3_client = get_s3_client()
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print(f"Arquivo {s3_key} excluído do bucket {bucket_name}")
    except ClientError as e:
        print(f"Erro ao excluir arquivo: {e}")

def delete_folder(bucket_name, folder_name):
    try:
        files = list_files(bucket_name, prefix=folder_name)
        for file in files:
            delete_file(bucket_name, file)
        print(f"Pasta {folder_name} excluída do bucket {bucket_name}")
    except ClientError as e:
        print(f"Erro ao excluir pasta: {e}")

def file_exists(bucket_name, s3_key):
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
    try:
        s3_client = get_s3_client()
        s3_client.delete_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} excluído com sucesso!")
    except ClientError as e:
        print(f"Erro ao excluir o bucket: {e}")

def empty_bucket(bucket_name):
    try:
        files = list_files(bucket_name)
        for file in files:
            delete_file(bucket_name, file)
        print(f"Bucket {bucket_name} limpo com sucesso!")
    except ClientError as e:
        print(f"Erro ao limpar o bucket: {e}")

def save_dataframe_to_parquet(df, bucket_name, s3_path):
    """
    Salva um DataFrame como um arquivo Parquet em um bucket do S3 com partição diária.

    :param df: DataFrame a ser salvo
    :param bucket_name: Nome do bucket no S3
    :param s3_path: Caminho no bucket S3 onde o arquivo deve ser salvo
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