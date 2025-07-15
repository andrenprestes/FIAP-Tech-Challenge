import json
import random

import json
import random

def criar_amostra_json_por_chaves(caminho_arquivo_entrada, caminho_arquivo_saida, tamanho_amostra=1000):
    """
    Cria uma amostra aleatória de registros de um arquivo JSON, mantendo a estrutura
    original de dicionário e as chaves dos registros selecionados.

    Args:
        caminho_arquivo_entrada (str): O caminho para o arquivo JSON de entrada.
        caminho_arquivo_saida (str): O caminho para o arquivo JSON de saída onde a amostra será salva.
        tamanho_amostra (int): O número de registros (pares chave-valor) a serem incluídos na amostra.
    """
    try:
        with open(caminho_arquivo_entrada, 'r', encoding='utf-8') as f_entrada:
            dados_brutos = json.load(f_entrada)

        # Verifica se os dados carregados são um dicionário
        if not isinstance(dados_brutos, dict):
            print("Erro: O arquivo JSON de entrada não contém um dicionário na raiz.")
            return

        # Extrai os itens (pares chave-valor) do dicionário principal
        # Isso nos dá uma lista de tuplas: [('5185', {data...}), ('5186', {data...}), ...]
        itens_registros = list(dados_brutos.items())

        # Garante que o tamanho da amostra não seja maior que o número total de registros
        if tamanho_amostra > len(itens_registros):
            print(f"Aviso: O tamanho da amostra ({tamanho_amostra}) é maior que o número total de registros ({len(itens_registros)}). "
                  "Todo o dicionário será incluído na amostra.")
            amostra_itens = itens_registros
        else:
            amostra_itens = random.sample(itens_registros, tamanho_amostra)

        # Reconstrói um novo dicionário a partir dos itens amostrados
        amostra_final = dict(amostra_itens)

        with open(caminho_arquivo_saida, 'w', encoding='utf-8') as f_saida:
            json.dump(amostra_final, f_saida, indent=4, ensure_ascii=False)

        print(f"Amostra de {len(amostra_final)} registros (com chaves) salva em '{caminho_arquivo_saida}' com sucesso!")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo_entrada}' não foi encontrado.")
    except json.JSONDecodeError:
        print(f"Erro: Não foi possível decodificar o arquivo JSON '{caminho_arquivo_entrada}'. Verifique a formatação.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


if __name__ == "__main__":
    # --- Configurações ---
    arquivo_json_entrada = "C:\\Users\\ffporto\\Desktop\\Estudo\\FIAP\\fase05\\data\\prospects.json"  # Nome do seu arquivo JSON de entrada
    arquivo_json_saida = "C:\\Users\\ffporto\\Desktop\\Estudo\\FIAP\\fase05\\data\\amostra_1000_prospects.json"  # Nome do arquivo JSON de saída para a amostra
    numero_de_registros = 1000             # Quantidade de registros na amostra

    # --- Exemplo de uso ---
    # Para testar, crie um arquivo JSON de exemplo chamado 'meu_arquivo.json'
    # com uma lista de objetos. Ex:
    """
    [
        {"id": 1, "nome": "Item A"},
        {"id": 2, "nome": "Item B"},
        ...
        {"id": 10000, "nome": "Item Z"}
    ]
    """

    criar_amostra_json_por_chaves(arquivo_json_entrada, arquivo_json_saida, numero_de_registros)