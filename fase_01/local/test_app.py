import unittest
from app import app


class APITestCase(unittest.TestCase):

    def setUp(self):
        # Configura o ambiente de teste
        app.config["TESTING"] = True
        app.config["JWT_SECRET_KEY"] = (
            "sua_chave_secreta_aqui"  # Certifique-se de que a chave secreta seja a mesma usada no app
        )
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

    def tearDown(self):
        # Limpeza após cada teste
        self.app_context.pop()

    def get_jwt_token(self):
        # Cria um token JWT para uso nos testes
        with app.test_client() as client:
            response = client.post(
                "/login", json={"username": "admin", "password": "senha123"}
            )
            self.assertEqual(response.status_code, 200)
            return response.json["access_token"]

    def test_login(self):
        # Teste para o endpoint de login
        response = self.app.post(
            "/login", json={"username": "admin", "password": "senha123"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json)

    def test_producao_csv_sem_token(self):
        # Teste sem JWT (deve falhar)
        response = self.app.get("/producaoCSV")
        self.assertEqual(response.status_code, 401)  # Falha por falta de token

    def test_producao_csv_com_token(self):
        # Teste com JWT válido
        token = self.get_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = self.app.get("/producaoCSV", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            isinstance(response.json, dict) or isinstance(response.json, list)
        )  # Verifica se o retorno é JSON

    def test_processamento_csv(self):
        # Teste para uma das rotas com parâmetro, como /processamentoCSV/tipo/Viniferas
        token = self.get_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = self.app.get("/processamentoCSV/tipo/Viniferas", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            isinstance(response.json, dict) or isinstance(response.json, list)
        )

    def test_importacao_csv(self):
        # Teste para a rota de importação com parâmetro /importacaoCSV/tipo/Vinhos
        token = self.get_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = self.app.get("/importacaoCSV/tipo/Vinhos", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            isinstance(response.json, dict) or isinstance(response.json, list)
        )

    def test_exportacao_csv(self):
        # Teste para a rota de exportação com parâmetro /exportacaoCSV/tipo/Vinho
        token = self.get_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = self.app.get("/exportacaoCSV/tipo/Vinho", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            isinstance(response.json, dict) or isinstance(response.json, list)
        )


if __name__ == "__main__":
    unittest.main()
