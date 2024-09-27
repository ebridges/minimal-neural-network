output "lambda_arn" {
  value = aws_lambda_function.mnist_lambda.arn
}

output "api_ui_endpoint" {
  value = "${aws_apigatewayv2_api.mnist_api.api_endpoint}/${aws_apigatewayv2_stage.mnist_stage.name}/ui"
}

output "api_urls_endpoint" {
  value = "${aws_apigatewayv2_api.mnist_api.api_endpoint}/${aws_apigatewayv2_stage.mnist_stage.name}/urls"
}

output "api_prediction_endpoint" {
  value = "${aws_apigatewayv2_api.mnist_api.api_endpoint}/${aws_apigatewayv2_stage.mnist_stage.name}/prediction"
}
