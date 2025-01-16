## API Flask com JWT - Execução Local com Docker

Esta API Flask permite a recuperação de dados CSV de diferentes categorias (produção, processamento, comercialização, importação e exportação), que são convertidos em JSON e retornados via rotas protegidas por JWT.

### Requisitos

- **Docker**: [Instalar Docker](https://docs.docker.com/get-docker/)

### Execução Local

1. **Clone o repositório**:

   ```bash
   git clone https://github.com/lis-r-barreto/FIAP-FASE-01.git
   cd FIAP-FASE-01
   ```

2. **Construa a imagem Docker**:

   ```bash
   docker build -t embrapa-api-flask .
   ```

3. **Execute o contêiner**:

   ```bash
   docker run -p 5000:5000 embrapa-api-flask
   ```

   A API estará disponível em `http://localhost:5000`.

### Testando a API

1. **Login para obter o token JWT**:

   ```bash
   curl -X POST http://localhost:5000/login \
   -H "Content-Type: application/json" \
   -d '{"username": "admin", "password": "senha123"}'
   ```

2. **Acessar endpoint protegido (exemplo: `/producaoCSV`)**:

   ```bash
   curl -X GET http://localhost:5000/producaoCSV \
   -H "Authorization: Bearer seu_token_jwt"
   ```

### Endpoints da API

- **`/login`**: Gera o token JWT para autenticação.
- **`/producaoCSV`**: Retorna dados de produção de uvas em formato JSON.
- **`/processamentoCSV/tipo/<tipo>`**: Retorna dados de processamento por tipo (Viníferas, Americanas, etc.).
- **`/comercializacaoCSV`**: Retorna dados de comercialização de uvas e derivados.
- **`/importacaoCSV/tipo/<tipo>`**: Retorna dados de importação por tipo (Vinhos, Espumantes, etc.).
- **`/exportacaoCSV/tipo/<tipo>`**: Retorna dados de exportação por tipo (Vinho, Espumantes, etc.).

### Tratamento de Erros

- **Falha na requisição HTTP**: Retorna uma mensagem de erro e código de status.
- **Exceções**: Em caso de exceção, retorna uma mensagem de erro com o código 500.

### Estrutura do Projeto

- **`Dockerfile`**: Configuração Docker para rodar a API.
- **`requirements.txt`**: Dependências necessárias para rodar o projeto, como Flask, pandas, requests e outras.

---
