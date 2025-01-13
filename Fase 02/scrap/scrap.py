from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from flask import jsonify
from datetime import datetime
from dotenv import load_dotenv
from s3_bucket_manager.manager import save_dataframe_to_parquet, ensure_bucket_exists, create_bucket
from .browser_detection import get_browser_driver
import os
import time
import pandas as pd

load_dotenv()

driver = get_browser_driver()

current_folder_path = os.getcwd()
download_dir = os.path.join(current_folder_path, "IBOVDia")

# URL da página
url = "https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=pt-br"

def get_ibov_data():
    """
    Scrapes IBOVESPA data from a specified webpage, processes it, and saves it to an S3 bucket in Parquet format.

    The function automates the process of navigating to a webpage, downloading a CSV file, 
    processing it into a DataFrame, and uploading the resulting data to an S3 bucket. It 
    also ensures the S3 bucket exists before saving the data.

    Steps:
        1. Navigate to the IBOVESPA data page.
        2. Click on the download button to retrieve a CSV file.
        3. Process the downloaded CSV file into a DataFrame.
        4. Ensure the specified S3 bucket exists or create it if necessary.
        5. Save the DataFrame to the S3 bucket in Parquet format with daily partitions.
        6. Remove the downloaded CSV file from the local system.

    Returns:
        flask.Response: A Flask JSON response with a success or error message and appropriate HTTP status code.

    Raises:
        Exception: Captures any errors during the scraping, file processing, or S3 operations, 
        returning an error message in the response.

    Notes:
        - The `download_dir` is defined as a subfolder named "IBOVDia" in the current working directory.
        - The `S3_BUCKET_NAME` is loaded from environment variables.
        - The function quits the WebDriver instance at the end, whether successful or not.
    """
    try:
        print("Iniciando o processo de coleta de dados...")
        driver.get(url)

        print("Downloading...")
        # Esperar até que o botão de download esteja visível e clicável
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//a[@href="/indexPage/day/IBOV?language=pt-br"]'))).click()

        time.sleep(10)  # Ajuste conforme necessário para garantir que o CSV seja baixado
        print("Processando...")
        csv_files = [f for f in os.listdir(download_dir) if f.startswith('IBOVDia_') and f.endswith('.csv')]

        if csv_files:
            print("Pegando ultimo arquivo...")
            latest_file = max(csv_files, key=lambda f: datetime.strptime(f, 'IBOVDia_%d-%m-%y.csv'))
            data = latest_file.split("_")[1].split(".")[0]  # Extração correta da data
            latest_file_path = os.path.join(download_dir, latest_file)

            print("Lendo como dataframe...")
            df = pd.read_csv(latest_file_path, encoding='ISO-8859-1', sep=';', header=1, index_col=None)
            df.reset_index(inplace=True)
            columns = df.columns[1:].to_list()
            columns.append("nan")
            df.columns = columns
            
            
            if df.iloc[:, -1].isna().all():
                df = df.iloc[:, :-1]

            # Adicionando coluna 'date' com partição diária
            df = df.iloc[:-2]
            print("Criando/verificando bucket...")

            # Criar/verificar bucket no S3
            bucket_name = os.getenv("S3_BUCKET_NAME")
            ensure_bucket_exists(bucket_name)


            save_dataframe_to_parquet(df, bucket_name, "raw")
            print(latest_file_path)
            os.remove(latest_file_path)

        else:
            return jsonify({"msg": "Nenhum arquivo CSV encontrado após download no site da IBOVESPA!"}), 500

    except Exception as e:
        return jsonify({"msg": f"Erro ao obter dados: {e}"}), 500

    finally:
        driver.quit()
        return jsonify({"msg": f"Dados salvos no S3 em formato Parquet com partições diárias!"}), 200
