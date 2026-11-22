output "public_ip" {
  description = "Fixed address of the server (repository variable EC2_HOST)"
  value       = aws_eip.rover.public_ip
}

output "dashboard" {
  value = "http://${aws_eip.rover.public_ip}:8080"
}

output "next_steps" {
  value = <<-EOT
    1. ssh ubuntu@${aws_eip.rover.public_ip}   (cloud-init needs ~3 min: cloud-init status --wait)
    2. cd /opt/rover && cp .env.example .env && nano .env
       (passwords, IMAGE_PREFIX=ghcr.io/<owner>/rover, COMPOSE_PROFILES=mlops or monitoring)
    3. GitHub -> Settings -> Variables: DEPLOY_TARGET=ec2, EC2_HOST=${aws_eip.rover.public_ip};
       Secrets: EC2_SSH_KEY = the key pair's private key
    4. Actions -> CD -> Run workflow (or push to main): deploys, smoke-tests, rolls back on failure
  EOT
}
