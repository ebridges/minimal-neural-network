variable "region" {
  description = "AWS region to deploy the resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket name to be used for storing MNIST images"
  type        = string
  default     = "com.eqbridges.mnist-img-archive"
}
