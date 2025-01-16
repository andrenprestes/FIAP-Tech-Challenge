import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from dotenv import load_dotenv
from s3_bucket_manager.manager import save_dataframe_to_parquet, ensure_bucket_exists
from .browser_detection import get_browser_driver
from pathlib import Path
import pandas as pd
import time

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do driver e diretório
driver = get_browser_driver()
download_dir = Path("IBOVdia").resolve()
download_dir.mkdir(exist_ok=True)

print(f"Diretório configurado: {download_dir}")

# URL da página
url = "https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=pt-br"

def transform_dataframe(df, data_str):
    """
    Aplica transformações no DataFrame para padronizar e preparar os dados.

    Args:
        df (pd.DataFrame): DataFrame lido do CSV.
        data_str (str): Data extraída do nome do arquivo para ser adicionada ao DataFrame.

    Returns:
        pd.DataFrame: DataFrame transformado e padronizado.
    """
    try:
        print("Transformando DataFrame...")

        df.reset_index(inplace=True)
        columns = df.columns
        columns = df.columns[1:].to_list()
        columns.append("nan")
        df.columns = columns
        if df.iloc[:, -1].isna().all():
            df = df.iloc[:, :-1]

        print("DataFrame inicial:")
        print(df.head(10))  # Mostrar os primeiros 10 registros para debug

        # Ajustar nomes das colunas
        df.columns = ["Codigo", "Acao", "Tipo", "Qtde_Teorica", "Part"]
        print("DataFrame após ajuste das colunas:")
        print(df.head(10))

        # Remover últimas linhas irrelevantes
        df = df.iloc[:-2]
        print("DataFrame após remoção de linhas irrelevantes:")
        print(df.tail(10))  # Mostrar as últimas 10 linhas

        # Adicionar coluna de data extraída do nome do arquivo
        df["Data"] = pd.to_datetime(data_str, format='%d-%m-%y')
        print("DataFrame após adição da coluna de data:")
        print(df.head(10))

        # Converter colunas numéricas
        df["Qtde_Teorica"] = df["Qtde_Teorica"].str.replace(".", "", regex=False).astype(float)
        df["Part"] = df["Part"].str.replace(",", ".", regex=False).astype(float)
        print("DataFrame após conversão de colunas numéricas:")
        print(df.head(10))

        return df

    except Exception as e:
        print(f"Erro ao transformar o DataFrame: {e}")
        raise


def process_csv_to_raw(csv_file_path, bucket_name):
    """
    Processa o arquivo CSV e salva os dados na camada raw do S3.
    """
    try:
        print(f"Lendo arquivo CSV: {csv_file_path}")

        # Extrair a data do nome do arquivo
        file_name = csv_file_path.name  # Nome do arquivo
        data_str = file_name.split("_")[1].split(".")[0]  # Extraindo a parte da data
        print(f"Data extraída do nome do arquivo: {data_str}")

        # Ler o arquivo CSV
        df = pd.read_csv(csv_file_path, encoding='ISO-8859-1', sep=';', header=1, index_col=0)
        print("DataFrame após leitura inicial:")
        print(df.head(10))  # Mostrar os primeiros 10 registros após a leitura

        # Transformar os dados
        df = transform_dataframe(df, data_str)

        # Garantir que o bucket existe
        ensure_bucket_exists(bucket_name)

        # Salvar no S3
        save_dataframe_to_parquet(df, bucket_name, s3_path="raw")
        print("Dados processados e salvos no S3 com sucesso!")

    except Exception as e:
        print(f"Erro ao processar o CSV: {e}")
        raise

def get_ibov_data():
    """
    Realiza o scrap dos dados da página da B3 e salva no S3.
    """
    try:
        print("Iniciando o processo de coleta de dados...")
        driver.get(url)

        # Remover arquivos antigos
        print("Limpando arquivos antigos...")
        for file in download_dir.iterdir():
            if file.suffix == ".csv":
                print(f"Removendo arquivo antigo: {file}")
                file.unlink()

        print("Tentando clicar no botão de download...")
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@href="/indexPage/day/IBOV?language=pt-br"]'))
        ).click()

        # Aguarde o download
        time.sleep(15)

        # Identificar o arquivo baixado
        csv_files = [f for f in download_dir.iterdir() if f.suffix == ".csv"]
        if not csv_files:
            print("Nenhum arquivo CSV encontrado no diretório.")
            return {"msg": "Nenhum arquivo CSV encontrado após download."}, 500

        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        print(f"Arquivo mais recente: {latest_file}")

        # Processar o arquivo para o S3
        process_csv_to_raw(latest_file, os.getenv("S3_BUCKET_NAME"))

        # Remover o arquivo local
        latest_file.unlink()
        print("Arquivo local removido.")

        return {"msg": "Dados processados e salvos com sucesso no S3."}, 200

    except Exception as e:
        print(f"Erro ao obter dados: {e}")
        return {"msg": f"Erro ao obter dados: {e}"}, 500

    finally:
        driver.quit()
