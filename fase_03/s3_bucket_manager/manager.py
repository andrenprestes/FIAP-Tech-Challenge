"""
Módulo para operações com o AWS S3 e manipulação de arquivos Parquet.

Este módulo fornece funções utilitárias para interagir com buckets do AWS S3,
realizar operações como upload, download, exclusão e listagem de arquivos,
bem como salvar um DataFrame em formato Parquet com partições diárias.
"""

from botocore.exceptions import NoCredentialsError, ClientError
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import boto3
from pathlib import Path
from io import BytesIO

def get_s3_client():
    """
    Retorna um cliente S3 usando boto3.
    """
    return boto3.client('s3')

def ensure_bucket_exists(bucket_name):
    """
    Verifica se um bucket S3 existe e cria caso não exista.
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} existe.")
    except ClientError:
        print(f"Bucket {bucket_name} não encontrado. Criando...")
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} criado com sucesso!")

def save_dataframe_to_parquet(df, bucket_name, s3_path):
    """
    Salva um DataFrame como Parquet no S3 com partições diárias.
    """
    partition_date = df['Data'].iloc[0].strftime('%Y%m%d') # Formatar data sem hífens

    def upload_to_s3(bucket_name, s3_path, partition_date, df, temp_file):
        """
        Faz upload de um DataFrame para o S3 no formato Parquet.
        """
        # Salvar o DataFrame como Parquet localmente com parâmetros de timestamp
        df.to_parquet(
            temp_file,
            engine="pyarrow",
            use_deprecated_int96_timestamps=True,
            allow_truncated_timestamps=True
        )

        # Caminho no S3
        file_name = f"ibov_{partition_date}.parquet"
        partitioned_path = f"{s3_path}/date={partition_date}/{file_name}"

        # Fazer upload para o S3
        s3_client = get_s3_client()
        print(f"Iniciando upload para: s3://{bucket_name}/{partitioned_path}")
        with open(temp_file, 'rb') as f:
            s3_client.put_object(Body=f.read(), Bucket=bucket_name, Key=partitioned_path)
        print(f"Arquivo salvo com sucesso: s3://{bucket_name}/{partitioned_path}")

    # Usar gerenciador de arquivos temporários para salvar o DataFrame
    def handle_temp_file(filename, operation_callback):
        temp_dir = Path("temp").resolve()
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / filename

        try:
            operation_callback(temp_file)
        except Exception as e:
            print(f"Erro ao operar com o arquivo temporário: {e}")
            raise
        finally:
            if temp_file.exists():
                temp_file.unlink()
                print(f"Arquivo temporário {temp_file} removido.")

    handle_temp_file(
        "data.parquet",
        lambda temp_file: upload_to_s3(bucket_name, s3_path, partition_date, df, temp_file)
    )

def load_parquet_from_s3(bucket_name, s3_path):
    """
    Carrega um arquivo Parquet de um bucket S3 em um DataFrame.
    """
    s3_client = get_s3_client()
    
    try:
        # Baixar o arquivo Parquet
        print(f"Baixando arquivo do S3: {s3_path}")
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
        parquet_data = response['Body'].read()

        # Carregar o Parquet em um DataFrame
        table = pq.read_table(BytesIO(parquet_data))
        df = table.to_pandas()
        print("Arquivo Parquet carregado com sucesso!")
        return df

    except Exception as e:
        print(f"Erro ao carregar o arquivo Parquet do S3: {e}")
        return pd.DataFrame()