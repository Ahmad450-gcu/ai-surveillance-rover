# The cloud deployment target: one EC2 server that runs the same docker-compose stack as the laptop.
#
#   cd deploy/aws
#   cp terraform.tfvars.example terraform.tfvars      # region, key pair, your IP
#   terraform init && terraform apply                 # ~2 min; prints the address and the next steps
#   terraform destroy                                 # when the demo is over (stops the cost)
#
# What it creates: a security group (SSH and the dashboard from your IP only), an Ubuntu 24.04 server with Docker
# installed by cloud-init (cloud-init.yaml) and the project cloned to /opt/rover, a fixed public IP (Elastic IP), and
# an encrypted disk. Credentials come from the AWS CLI / environment (AWS_PROFILE or AWS_ACCESS_KEY_ID), never
# from files in the repository.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project   = "ai-surveillance-rover"
      ManagedBy = "terraform"
    }
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "rover" {
  name        = "${var.name}-sg"
  description = "Rover stack: SSH and the dashboard from the admin address only"

  ingress {
    description = "SSH (deployments from GitHub Actions use it too)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_cidrs
  }

  dynamic "ingress" {
    for_each = var.open_ports
    content {
      description = "service port ${ingress.value}"
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = var.app_cidrs
    }
  }

  egress {
    description = "updates, image pulls, git"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "rover" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.rover.id]

  root_block_device {
    volume_type = "gp3"
    volume_size = var.disk_gb
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  user_data = templatefile("${path.module}/cloud-init.yaml", {
    repo_url    = var.repo_url
    deploy_user = "ubuntu"
    swap_gb     = var.swap_gb
  })
  user_data_replace_on_change = false

  tags = {
    Name = var.name
  }
}

resource "aws_eip" "rover" {
  instance = aws_instance.rover.id
  domain   = "vpc"
  tags = {
    Name = "${var.name}-ip"
  }
}
