import boto3

def lambda_handler(event, context):
    glue_job_name = "Glue TECH CHALLENGE 2 "
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