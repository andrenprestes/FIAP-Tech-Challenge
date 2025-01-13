# import boto3
# from botocore.exceptions import NoCredentialsError

# # Inicializa o cliente S3
# s3 = boto3.client('s3')

# # Nome do bucket (deve ser único globalmente)
# bucket_name = 'raw-data-bucket-bovespa'
# region = 'us-east-1'  # escolha a região desejada

# try:
#     # Cria o bucket
#     s3.create_bucket(
#         Bucket=bucket_name
#     )
#     print(f'Bucket {bucket_name} criado com sucesso!')
# except NoCredentialsError:
#     print('Credenciais não encontradas. Verifique suas chaves de acesso.')
# except Exception as e:
#     print(f'Erro ao criar o bucket: {e}')

import boto3

def list_glue_jobs():
    """
    Lists all jobs available in AWS Glue and prints their names.

    This function initializes a Glue client using the boto3 library, retrieves 
    the list of all job names configured in the AWS Glue service, and prints 
    them to the console. If no jobs are found, it displays a message indicating 
    that no jobs exist. If an error occurs during the process, the function 
    handles the exception and logs an error message.

    Returns:
        list: A list of job names (strings) from AWS Glue. Returns an empty 
        list if no jobs are found or if an error occurs.

    Raises:
        Exception: Captures and logs any exception raised during the 
        communication with the AWS Glue service.

    Dependencies:
        - boto3: Ensure the AWS SDK for Python is installed and properly 
          configured with the necessary credentials and permissions to access 
          AWS Glue.
          
    Notes:
        - The function uses the default AWS credentials and region configured 
          in the environment or AWS configuration files.
        - Ensure that the IAM role or user has permissions to access the 
          `glue:ListJobs` API.

    """
    # Inicializar o cliente do Glue
    glue_client = boto3.client('glue')

    try:
        # Listar todos os jobs
        response = glue_client.list_jobs()

        # Obter a lista de nomes de jobs
        job_names = response.get('JobNames', [])

        if job_names:
            print("Jobs no Glue:")
            for job in job_names:
                print(f"- {job}")
        else:
            print("Nenhum job encontrado no Glue.")

        return job_names
    except Exception as e:
        print(f"Erro ao listar os jobs: {str(e)}")
        return []

# Executar a função
list_glue_jobs()
