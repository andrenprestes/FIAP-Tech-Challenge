terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "5.17.0" }
  }
}

provider "aws" {
  region  = "us-east-1"
}

# --- ECR ---

resource "aws_ecr_repository" "embrapa_api" {
  name                 = "lambda-embrapa-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle {
    prevent_destroy = false
  }
}

# --- Build & push image ---

locals {
  repo_url = aws_ecr_repository.embrapa_api.repository_url
}

resource "null_resource" "image" {
  triggers = {
    hash = md5(join("-", [for x in fileset("", "./deployment/{*.py,*.txt,Dockerfile}") : filemd5(x)]))
  }

  provisioner "local-exec" {
    command = <<EOF
      aws ecr get-login-password | docker login --username AWS --password-stdin ${local.repo_url}
      docker build --platform linux/amd64 -t ${local.repo_url}:latest ./deployment
      docker push ${local.repo_url}:latest
    EOF
  }
}

data "aws_ecr_image" "latest" {
  repository_name = aws_ecr_repository.embrapa_api.name
  image_tag       = "latest"
  depends_on      = [null_resource.image]
}

# --- IAM Role ---

resource "aws_iam_role" "embrapa_lambda" {
  name = "embrapa_lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })

  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.embrapa_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_full_access" {
  role       = aws_iam_role.embrapa_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/AWSLambda_FullAccess"
}

resource "aws_iam_role_policy_attachment" "ecr_full_access" {
  role       = aws_iam_role.embrapa_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
}

# --- CloudWatch Logs ---

resource "aws_cloudwatch_log_group" "embrapa_api" {
  name              = "/aws/lambda/embrapa_api"
  retention_in_days = 14

  lifecycle {
    prevent_destroy = false
  }
}

# --- Lambda ---

resource "aws_lambda_function" "embrapa_api" {
  function_name    = "embrapa_api"
  role             = aws_iam_role.embrapa_lambda.arn
  image_uri        = "${aws_ecr_repository.embrapa_api.repository_url}:latest"
  package_type     = "Image"
  source_code_hash = trimprefix(data.aws_ecr_image.latest.id, "sha256:")
  timeout          = 10

  environment {
    variables = {}
  }

  depends_on = [
    null_resource.image,
    aws_iam_role_policy_attachment.lambda_logs,
    aws_cloudwatch_log_group.embrapa_api,
  ]
}

# --- Lambda Endpoint ---

resource "aws_lambda_function_url" "embrapa_api" {
  function_name      = aws_lambda_function.embrapa_api.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = true
    allow_origins     = ["*"]
    allow_methods     = ["*"]
    allow_headers     = ["date", "keep-alive"]
    expose_headers    = ["keep-alive", "date"]
    max_age           = 86400
  }
}

output "api_url" {
  value = aws_lambda_function_url.embrapa_api.function_url
}
