from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from data_handler import initialize_historical_data, update_historical_data, fetch_realtime_price

app = Flask(__name__)

# Configuração do JWT
app.config['JWT_SECRET_KEY'] = 'sua_chave_secreta_aqui'  # Defina uma chave secreta forte
jwt = JWTManager(app)

# Variáveis globais
historical_data = initialize_historical_data()

@app.route('/login', methods=['POST'])
def login():
    """Endpoint para autenticação de usuário e geração de um token de acesso JWT."""
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)

    if username != 'admin' or password != 'senha123':
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/current_value', methods=['GET'])
@jwt_required()
def get_current_value():
    """Endpoint protegido para obter o valor atual do Bitcoin."""
    price = fetch_realtime_price()
    return jsonify({"realtime_price": price}), 200

@app.route('/historical', methods=['GET'])
@jwt_required()
def get_historical():
    """Endpoint protegido para obter os dados históricos do Bitcoin."""
    return jsonify(historical_data.to_dict(orient="records")), 200

@app.route('/update_historical', methods=['POST'])
@jwt_required()
def update_historical():
    """Endpoint para atualizar os dados históricos."""
    global historical_data
    historical_data = update_historical_data(historical_data)
    return jsonify({"msg": "Dados históricos atualizados com sucesso!"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
