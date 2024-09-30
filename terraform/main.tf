provider "aws" {
  region = var.region
}

# Define S3 bucket for Lambda deployment package
data "aws_s3_bucket" "lambda_bucket" {
  bucket = var.bucket_name
}

# Create the Lambda role with GetObject permission for S3
resource "aws_iam_role" "lambda_execution_role" {
  name = "lambda_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Attach S3 permission policy
resource "aws_iam_policy" "lambda_s3_policy" {
  name        = "LambdaS3Policy"
  description = "Allow Lambda to generate presigned URLs for S3 objects"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "s3:GetObject"
        ],
        Resource = [
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      }
    ]
  })
}

# Attach policies to the Lambda role
resource "aws_iam_role_policy_attachment" "lambda_s3_policy_attachment" {
  role       = aws_iam_role.lambda_execution_role.name
  policy_arn = aws_iam_policy.lambda_s3_policy.arn
}

resource "aws_iam_role_policy_attachment" "lambda_basic_policy_attachment" {
  role       = aws_iam_role.lambda_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Package and deploy the Lambda function
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../server"  # Path to your Python code
  output_path = "${path.module}/lambda.zip"
}

resource "aws_lambda_function" "mnist_lambda" {
  function_name = "mnist-predictor"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "prediction_service.lambda_handler"
  runtime       = "python3.11"

  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = filebase64sha256(data.archive_file.lambda_zip.output_path)

  environment {
    variables = {
      S3_BUCKET = var.bucket_name
    }
  }
  timeout = 10
}

# Create API Gateway HTTP API
resource "aws_apigatewayv2_api" "mnist_api" {
  name          = "MNIST Image Grid API"
  protocol_type = "HTTP"
}

# Create a route for the Lambda function
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                = aws_apigatewayv2_api.mnist_api.id
  integration_type      = "AWS_PROXY"
  integration_uri       = aws_lambda_function.mnist_lambda.invoke_arn
  payload_format_version = "2.0"
}

# Create route for "GET /urls"
resource "aws_apigatewayv2_route" "mnist_route_urls" {
  api_id    = aws_apigatewayv2_api.mnist_api.id
  route_key = "GET /urls"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# Create route for "GET /ui"
resource "aws_apigatewayv2_route" "mnist_route_ui" {
  api_id    = aws_apigatewayv2_api.mnist_api.id
  route_key = "GET /ui"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# Create route for "POST /prediction"
resource "aws_apigatewayv2_route" "mnist_route_prediction" {
  api_id    = aws_apigatewayv2_api.mnist_api.id
  route_key = "POST /prediction/{filename}"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "mnist_stage" {
  api_id      = aws_apigatewayv2_api.mnist_api.id
  name        = "dev"
  auto_deploy = true
}

# Allow API Gateway to invoke the Lambda
resource "aws_lambda_permission" "apigateway_lambda_permission" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.mnist_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.mnist_api.execution_arn}/*"
}
