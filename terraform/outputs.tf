output "lambda_arn" {
  value = aws_lambda_function.mnist_lambda.arn
}

output "api_endpoint" {
  value = "${aws_apigatewayv2_api.mnist_api.api_endpoint}/${aws_apigatewayv2_stage.mnist_stage.name}/mnist-urls"
}
