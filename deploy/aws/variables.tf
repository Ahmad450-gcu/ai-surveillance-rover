variable "region" {
  description = "AWS region, e.g. me-central-1 (UAE) or eu-central-1"
  type        = string
  default     = "eu-central-1"
}

variable "name" {
  description = "Name of the server and its resources"
  type        = string
  default     = "rover-prod"
}

variable "instance_type" {
  description = "t3.large (8 GB): core + mlops or monitoring; t3.xlarge (16 GB): every profile at once"
  type        = string
  default     = "t3.large"
}

variable "disk_gb" {
  description = "Root disk: images (~8 GB), lake, database, Airflow logs"
  type        = number
  default     = 60
}

variable "swap_gb" {
  description = "Swap file size (headroom while a retraining runs)"
  type        = number
  default     = 4
}

variable "key_name" {
  description = "Name of an existing EC2 key pair (its private key is the GitHub secret EC2_SSH_KEY)"
  type        = string
}

variable "ssh_cidrs" {
  description = "Who may SSH in: your IP (x.x.x.x/32); for deployments from GitHub-hosted runners see docs/DEPLOYMENT.md"
  type        = list(string)
}

variable "app_cidrs" {
  description = "Who may open the service ports (dashboard, Grafana, Airflow)"
  type        = list(string)
}

variable "open_ports" {
  description = "Service ports opened to app_cidrs: 8080 dashboard/API; add 3000 Grafana, 8081 Airflow, 5000 MLflow if needed"
  type        = list(number)
  default     = [8080]
}

variable "repo_url" {
  description = "Git repository cloned to /opt/rover"
  type        = string
  default     = "https://github.com/Ahmad450-gcu/ai-surveillance-rover.git"
}
