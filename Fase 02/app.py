from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask import Flask, jsonify, request
import scrap.scrap as scrap

app = Flask(__name__)

# Configuração do JWT
app.config['JWT_SECRET_KEY'] = 'sua_chave_secreta_aqui'  # Defina uma chave secreta forte
jwt = JWTManager(app)

port = 5000


# Função para criar token de acesso
@app.route('/login', methods=['POST'])
def login():
    """
    Endpoint para autenticação de usuário e geração de um token de acesso JWT.

    Este endpoint valida as credenciais do usuário recebidas em formato JSON, verificando se o 'username'
    e 'password' correspondem às credenciais esperadas. Caso a autenticação seja bem-sucedida, um token
    JWT é gerado e retornado para que o usuário possa acessar outros endpoints protegidos da API.

    Request Body (JSON):
        - username (str): O nome de usuário do cliente.
        - password (str): A senha do cliente.

    Returns:
        JSON:
            - Se o JSON estiver ausente ou incorreto, retorna:
                {
                    "msg": "Missing JSON in request"
                }, com status HTTP 400 (Bad Request).

            - Se o 'username' ou 'password' forem inválidos, retorna:
                {
                    "msg": "Bad username or password"
                }, com status HTTP 401 (Unauthorized).

            - Se a autenticação for bem-sucedida, retorna:
                {
                    "access_token": "<token>"
                }, com status HTTP 200 (OK).
    """
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)

    if username != 'admin' or password != 'senha123':
        return jsonify({"msg": "Bad username or password"}), 401

    # Cria o token de acesso
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/bovespaDay')
@jwt_required()
def path_get_ibov_data():
    """
    Função para obter os dados diários do índice IBOVESPA e salvá-los em formato Parquet.

    Essa função utiliza o Selenium para acessar o site oficial da B3, localizar o botão de download do arquivo CSV contendo os dados diários do índice IBOVESPA, e processar o arquivo baixado para salvar os dados em formato Parquet.

    Passos executados:
        1. Acessa o site da B3 usando o Selenium em modo headless.
        2. Localiza e clica no botão de download do arquivo CSV.
        3. Aguarda o download do arquivo.
        4. Processa o arquivo CSV:
            - Renomeia as colunas para um formato mais adequado.
            - Adiciona uma coluna de data com base no nome do arquivo.
            - Converte a coluna de data para o tipo datetime.
        5. Salva os dados em um arquivo Parquet no diretório especificado.
        6. Remove o arquivo CSV original após o processamento.

    Returns:
        JSON:
            - Se o botão de download não for encontrado ou estiver desativado, retorna:
                {
                    "msg": "Botão de download não encontrado ou desativado."
                }, com status HTTP 500 (Internal Server Error).
            
            - Se nenhum arquivo CSV for encontrado após o download, retorna:
                {
                    "msg": "Nenhum arquivo CSV encontrado após download no site da IBOVESPA!"
                }, com status HTTP 500 (Internal Server Error).
            
            - Em caso de erro durante a execução, retorna:
                {
                    "msg": "Erro ao obter dados: <detalhes do erro>"
                }, com status HTTP 500 (Internal Server Error).

    Observações:
        - O tempo de espera para carregamento da página e download do arquivo pode variar dependendo da velocidade da conexão e do site.
        - Certifique-se de que o caminho para o driver do Selenium e o diretório de download estão corretos.
        - O arquivo Parquet é particionado por data para facilitar futuras consultas.

    Requisitos:
        - Selenium WebDriver configurado para o navegador Firefox.
        - O driver `geckodriver` deve estar no caminho especificado.
        - Diretório de download configurado para salvar arquivos CSV.

    """
    return scrap.get_ibov_data()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)
