from flask import Flask, jsonify, request
import pandas as pd
import requests
from io import StringIO
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from mangum import Mangum
from asgiref.wsgi import WsgiToAsgi

app = Flask(__name__)
app.config["DEBUG"] = True
asgi_app = WsgiToAsgi(app)
handler = Mangum(asgi_app)

# Configuração do JWT
app.config["JWT_SECRET_KEY"] = (
    "sua_chave_secreta_aqui"  # Defina uma chave secreta forte
)
jwt = JWTManager(app)

# URL base para download de CSV
url_base_csv = "http://vitibrasil.cnpuv.embrapa.br/download/"


# Função para criar token de acesso
@app.route("/login", methods=["POST"])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get("username", None)
    password = request.json.get("password", None)

    # Verificação simples de usuário/senha
    if username != "admin" or password != "senha123":
        return jsonify({"msg": "Bad username or password"}), 401

    # Cria o token de acesso
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)


# Endpoints protegidos usando JWT
@app.route("/producaoCSV")
@jwt_required()
def producao_csv():
    return get_data("Producao")


@app.route("/processamentoCSV/tipo/<tipo>")
@jwt_required()
def processamento_csv(tipo):
    return get_data(f"Processa{tipo}")


@app.route("/comercializacaoCSV")
@jwt_required()
def comercializacao_csv():
    return get_data("Comercio")


@app.route("/importacaoCSV/tipo/<tipo>")
@jwt_required()
def importacao_csv(tipo):
    return get_data(f"Imp{tipo}")


@app.route("/exportacaoCSV/tipo/<tipo>")
@jwt_required()
def exportacao_csv(tipo):
    return get_data(f"Exp{tipo}")


def get_data(page: str):
    try:
        response = requests.get(url_base_csv + page + ".csv")
        if response.status_code == 200:
            csv_data = StringIO(response.content.decode("utf-8"))
            df = pd.read_csv(csv_data, sep=";")
            df.fillna(0, inplace=True)

            if "control" in df.columns:
                df["control"] = df["control"].astype(str)
                mask_upper = df["control"].str.isupper()

                df_dict = {}
                indices = df[mask_upper].index.tolist()
                indices.append(len(df))

                if not indices or len(indices) == 1:
                    return jsonify(df.to_dict(orient="records"))

                for i in range(len(indices) - 1):
                    df_temp = df.iloc[indices[i]:indices[i + 1]]
                    key = df_temp.iloc[0]["control"]
                    df_temp = df_temp.drop(columns=["control"])
                    df_dict[key] = df_temp.to_dict(orient="records")

                return jsonify(df_dict)
            else:
                return jsonify(df.to_dict(orient="records"))
        else:
            return (
                jsonify(
                    {
                        "error": f"Failed to retrieve data, status code: {response.status_code}"
                    }
                ),
                response.status_code,
            )

    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

