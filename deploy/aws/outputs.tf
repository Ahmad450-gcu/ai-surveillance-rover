output "public_ip" {
  description = "Fixed address of the server (repository variable EC2_HOST)"
  value       = aws_eip.rover.public_ip
}

output "instance_id" {
  description = "The server (tools/aws.py start / stop / status)"
  value       = aws_instance.rover.id
}

output "region" {
  value = var.region
}

output "dashboard" {
  value = "http://${aws_eip.rover.public_ip}:8080"
}

output "next_steps" {
  value = <<-EOT
    1. python tools/aws.py deploy       installs the latest published release and tests it (~10 min)
    2. python tools/aws.py copy-data    this computer's alerts, models and training data (Docker must run here)
    3. python tools/aws.py tunnel       every service on localhost
    Automatic deployments: GitHub -> Settings -> Variables: DEPLOY_TARGET=ec2, EC2_HOST=${aws_eip.rover.public_ip};
    Secrets: EC2_SSH_KEY = the private key (~/.ssh/rover_aws)
  EOT
}
