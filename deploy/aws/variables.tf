variable "region" {
  description = "AWS region (the new-style Free plan accounts have exactly one; from Pakistan: Sydney)"
  type        = string
  default     = "ap-southeast-2"
}

variable "name" {
  description = "Name of the server and its resources"
  type        = string
  default     = "rover-prod"
}

variable "instance_type" {
  description = "m7i-flex.large: 2 vCPU, 8 GB, Free plan eligible; t3.xlarge (16 GB) needs a Paid plan"
  type        = string
  default     = "m7i-flex.large"
}

variable "disk_gb" {
  description = "Root disk: images (~10 GB per release), lake, database, training data, Airflow logs"
  type        = number
  default     = 60
}

variable "swap_gb" {
  description = "Swap file size (headroom while a retraining runs)"
  type        = number
  default     = 4
}

variable "public_key_file" {
  description = "Public SSH key put on the server (its private key is the GitHub secret EC2_SSH_KEY)"
  type        = string
  default     = "~/.ssh/rover_aws.pub"
}

variable "ssh_cidrs" {
  description = "Who may reach SSH (the server accepts the key only); GitHub's runners have changing addresses"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allow_my_ip" {
  description = "Open the service ports to the address Terraform runs from"
  type        = bool
  default     = true
}

variable "app_cidrs" {
  description = "More addresses that may open the service ports, e.g. the university's"
  type        = list(string)
  default     = []
}

variable "open_ports" {
  description = "Ports opened to those addresses: 8080 dashboard/API, 3000 Grafana (the rest: python tools/aws.py tunnel)"
  type        = list(number)
  default     = [8080, 3000]
}

variable "timezone" {
  description = "The server's time zone (the daily stop uses it)"
  type        = string
  default     = "Asia/Karachi"
}

variable "stop_at" {
  description = "The server stops itself every day at this time (HH:MM), so a forgotten server costs at most one day"
  type        = string
  default     = "23:30"
  validation {
    condition     = can(regex("^([01][0-9]|2[0-3]):[0-5][0-9]$", var.stop_at))
    error_message = "stop_at must be HH:MM, e.g. 23:30."
  }
}

variable "repo_url" {
  description = "Git repository cloned to /opt/rover"
  type        = string
  default     = "https://github.com/Ahmad450-gcu/ai-surveillance-rover.git"
}
