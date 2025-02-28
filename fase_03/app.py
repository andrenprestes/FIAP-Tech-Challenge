from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask import Flask, jsonify, request
from coinGecko.fill_db import fill_db, get_curret_data

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

@app.route('/preenche_db')
@jwt_required()
def path_coingecko_fill_db():
    """
    Endpoint protegido para preencher o banco de dados com dados obtidos do CoinGecko.

    Este endpoint exige um token de acesso JWT válido para ser acessado. Quando chamado, ele preenche o banco
    de dados com os dados mais recentes do CoinGecko, provavelmente envolvendo a coleta de cotações de criptomoedas
    e o armazenamento dessas informações no banco.

    Headers:
        Authorization: Bearer <token>

    Returns:
        JSON:
            - Retorna uma mensagem de sucesso ou erro, dependendo da operação realizada no banco.
            - Em caso de erro, o status HTTP será 500 (Internal Server Error).
    """
    fill_db()

@app.route('/current_value')
@jwt_required()
def get_current_value():
    """
    Endpoint protegido para obter o valor atual de uma criptomoeda.

    Este endpoint exige um token de acesso JWT válido para ser acessado. Quando chamado, ele retorna o valor atual
    de uma criptomoeda, como o Bitcoin, no formato JSON.

    Headers:
        Authorization: Bearer <token>

    Returns:
        JSON:
            - Retorna o valor atual da criptomoeda solicitada, como o Bitcoin, no formato:
            {
                "price": <current_price>
            }, com status HTTP 200 (OK).

            - Em caso de falha ao obter os dados, retorna um erro com status HTTP 500 (Internal Server Error).
    """
    get_curret_data()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)
