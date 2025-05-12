import requests
import pandas as pd
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone

def save_dataframe_to_parquet(df, local_dir):
    """
    Salva um DataFrame como Parquet localmente com partições diárias.
    """
    partition_date = df['date'].iloc[0].strftime('%Y%m%d')  # Formatar data sem hífens
    file_name = f"ibov_{partition_date}.parquet"
    partitioned_path = Path(local_dir) / f"date={partition_date}" / file_name

    # Criar diretório se não existir
    partitioned_path.parent.mkdir(parents=True, exist_ok=True)

    # Salvar o DataFrame como Parquet
    df.to_parquet(
        partitioned_path,
        engine="pyarrow",
        use_deprecated_int96_timestamps=True,
        allow_truncated_timestamps=True
    )

    print(f"Arquivo salvo localmente em: {partitioned_path}")

def get_historic(crypto_id="bitcoin", currency="usd", days=365):
    """
    Obtém o histórico de preços da criptomoeda utilizando a API do CoinGecko.

    Parâmetros:
        - crypto_id (str): O ID da criptomoeda (padrão: 'bitcoin').
        - currency (str): A moeda em que a criptomoeda será cotada (padrão: 'usd').
        - days (int): O número de dias históricos a serem obtidos (padrão: 365).

    Retorna:
        - DataFrame: Contendo os dados de preço e data da criptomoeda.
    """
    try:
        # Configurações da URL da API CoinGecko
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {"vs_currency": currency, "days": days}

        # Fazendo a requisição
        response = requests.get(url, params=params)
        response.raise_for_status()  # Levanta exceções para status HTTP 4xx/5xx

        # Processa os dados retornados
        data = response.json()

        # Verifica se a chave 'prices' está presente
        if "prices" not in data:
            raise ValueError("Dados de preços não encontrados na resposta da API.")

        prices = data["prices"]  # Lista de timestamps e preços
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        print(df.head())
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        save_dataframe_to_parquet(df, r"C:\Users\ffporto\PycharmProjects\FIAP-Tech-Challenge\fase_03")

        return "Passou"
    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar a API do CoinGecko: {e}")
        return None
    except ValueError as e:
        print(f"Erro de dados: {e}")
        return None
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None

get_historic()