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