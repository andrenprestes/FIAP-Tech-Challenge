import boto3

def lambda_handler(event, context):
    """
    Função Lambda para verificar e iniciar um job do AWS Glue.

    Esta função verifica a existência de um job do AWS Glue com o nome especificado 
    e, se encontrado, inicia a execução do job. Caso o job não seja encontrado ou ocorra 
    algum erro, a função retorna mensagens apropriadas.

    Parâmetros:
        event (dict): Dados do evento que acionam a função Lambda (não utilizado neste caso).
        context (object): Contexto de execução da função Lambda (contém informações como ID de execução).

    Retorno:
        dict: Um dicionário contendo:
            - statusCode (int): Código de status HTTP indicando o resultado da operação.
            - body (str): Mensagem explicativa sobre o resultado da operação.

    Respostas possíveis:
        - statusCode: 200
          body: "Job iniciado com sucesso. JobRunId: <JobRunId>"
          Indica que o job foi encontrado e iniciado com sucesso.

        - statusCode: 404
          body: "Job '<job_name>' não encontrado. Verifique se o nome está correto."
          Indica que o job especificado não existe.

        - statusCode: 500
          body: "Erro ao iniciar o job: <mensagem de erro>"
          Indica que ocorreu um erro inesperado durante a execução.

    Exceções tratadas:
        - glue_client.exceptions.EntityNotFoundException:
            Disparada quando o job especificado não é encontrado no Glue.
        - Exception:
            Captura erros genéricos que podem ocorrer durante a execução da função.

    Dependências:
        - boto3: Biblioteca necessária para interagir com os serviços da AWS.
        - Configuração adequada de permissões IAM para permitir acesso ao AWS Glue.

    Observação:
        Certifique-se de que o nome do job no código corresponde exatamente ao nome do job configurado no AWS Glue.
    """
    glue_job_name = "glue-job-batch-bovespa"
    glue_client = boto3.client('glue')
    
    try:
        # First check if the job exists
        glue_client.get_job(JobName=glue_job_name)
        
        # If job exists, start it
        response = glue_client.start_job_run(
            JobName=glue_job_name
        )
        
        return {
            'statusCode': 200,
            'body': f"Job iniciado com sucesso. JobRunId: {response['JobRunId']}"
        }
    
    except glue_client.exceptions.EntityNotFoundException:
        return {
            'statusCode': 404,
            'body': f"Job '{glue_job_name}' não encontrado. Verifique se o nome está correto."
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f"Erro ao iniciar o job: {str(e)}"
        }
