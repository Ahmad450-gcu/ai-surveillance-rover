# The cloud deployment target: one EC2 server that runs the same docker-compose stack as the laptop.
#
#   python tools/aws.py prepare          # SSH key and the server's passwords (once)
#   python tools/aws.py plan             # what would be created; changes nothing
#   python tools/aws.py apply            # asks, then creates it (~2 min)
#   python tools/aws.py destroy          # removes everything (stops the cost)
# (or terraform init / plan / apply in this folder with AWS credentials set)
#
# What it creates: a key pair, a security group (SSH with the key; the dashboard and Grafana from your IP only), an
# Ubuntu 24.04 server with Docker installed by cloud-init (cloud-init.yaml) and the project cloned to /opt/rover,
# a fixed public IP (Elastic IP), and an encrypted disk. Every other service is reached through an SSH tunnel
# (python tools/aws.py tunnel). Credentials come from the AWS CLI ("aws login") / environment, never from files in
# the repository.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.4"
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

# The address this computer has on the internet right now: the dashboard opens only to it (and var.app_cidrs).
# Home connections change address now and then: "python tools/aws.py apply" again updates the rule.
data "http" "my_ip" {
  count = var.allow_my_ip ? 1 : 0
  url   = "https://checkip.amazonaws.com"
}

locals {
  my_cidr   = var.allow_my_ip ? ["${chomp(data.http.my_ip[0].response_body)}/32"] : []
  app_cidrs = distinct(concat(local.my_cidr, var.app_cidrs))
}

resource "aws_key_pair" "rover" {
  key_name   = "${var.name}-key"
  public_key = file(pathexpand(var.public_key_file))
}

resource "aws_security_group" "rover" {
  name        = "${var.name}-sg"
  description = "Rover stack: SSH by key, the dashboard from the admin address only"

  ingress {
    description = "SSH, keys only (deployments from GitHub Actions use it too)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_cidrs
  }

  dynamic "ingress" {
    for_each = length(local.app_cidrs) > 0 ? var.open_ports : []
    content {
      description = "service port ${ingress.value}"
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = local.app_cidrs
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
  key_name               = aws_key_pair.rover.key_name
  vpc_security_group_ids = [aws_security_group.rover.id]

  # the daily "shutdown" on the server (cloud-init.yaml) stops it: no hourly cost, the disk and its data stay
  instance_initiated_shutdown_behavior = "stop"

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
    timezone    = var.timezone
    stop_at     = var.stop_at
  })
  user_data_replace_on_change = false

  lifecycle {
    ignore_changes = [ami, user_data] # a newer Ubuntu image must never replace the server (and its data)
  }

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
