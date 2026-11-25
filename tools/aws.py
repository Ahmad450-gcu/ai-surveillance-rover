"""The AWS server, from this computer (docs/DEPLOYMENT.md). Needs the AWS CLI, Terraform and OpenSSH.

    once      aws login                          sign in (browser)
              python tools/aws.py check          account, region, server type, limits; creates nothing
              python tools/aws.py prepare        SSH key + the server's own passwords (deploy/aws/server.env, MY_LOGINS.txt)
              python tools/aws.py plan           what Terraform would create; creates nothing
              python tools/aws.py apply          creates the server (asks first). Again later = open the dashboard to
                                                 this computer's new internet address
              python tools/aws.py deploy         installs the latest published release and tests it
              python tools/aws.py copy-data      this computer's alerts, models and training data -> the server

    or        python tools/aws.py user-data      a server made by hand in the AWS console: the file for its "User data" box
              python tools/aws.py use <ip> [instance-id]    ... and its address, instead of plan / apply

    daily     python tools/aws.py start          switch on; prints the addresses
              python tools/aws.py tunnel         every service on localhost until Ctrl+C
              python tools/aws.py run -- --source clip.mp4 --set streaming.camera_id=pi-cam
                                                 the detector here (run.py), everything else on the server
              python tools/aws.py status         on / off, containers, memory, credits left
              python tools/aws.py model rollback 1   the model registry on the server (deploy/model.sh: list,
                                                 promote, rollback <version>)
              python tools/aws.py stop           switch off (it also stops itself every night)

    the end   python tools/aws.py destroy        deletes the server and its data (asks first)
"""
import base64
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / "deploy" / "aws"
ENV_FILE = TF / "server.env"
MANUAL = TF / "server.json"                                   # a server made by hand (command "use")
USER_DATA = TF / "user-data.yaml"
LOGINS = ROOT / "MY_LOGINS.txt"
KEY = Path.home() / ".ssh" / "rover_aws"
REMOTE = "/opt/rover"
ALL_PROFILES = "streaming,monitoring,mlops"

# services on localhost while the tunnel is open: the same ports as the local stack, so every link and tool works
PORTS = {"dashboard": 8080, "Grafana": 3000, "Airflow": 8081, "MLflow": 5000, "MinIO": 9001, "MinIO API": 9000,
         "Spark": 4040, "Prometheus": 9090, "Alertmanager": 9093, "drift report": 9200, "serving": 8000, "Kafka": 9094}
RUN_PORTS = ("dashboard", "Kafka", "serving", "MLflow")      # what the detector needs
LIVE_VIEW_PORT = 8090                                         # 8081 is Airflow when a tunnel is open
# everything that uses the databases (stopped while they are replaced)
DB_USERS = ("backend mlflow serving spark alert-bridge grafana drift-monitor airflow-scheduler airflow-webserver")


def program(name):
    """Full path; freshly installed programs are not on the PATH of an already open terminal."""
    found = shutil.which(name)
    if found:
        return found
    guesses = [Path(r"C:\Program Files\Amazon\AWSCLIV2\aws.exe")] if name == "aws" else []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        guesses.append(Path(local) / "Microsoft" / "WinGet" / "Links" / f"{name}.exe")
        guesses += sorted((Path(local) / "Microsoft" / "WinGet" / "Packages").glob(f"Hashicorp.{name}_*/{name}.exe"))
    for g in guesses:
        if g.exists():
            return str(g)
    sys.exit(f"{name} is not installed (docs/DEPLOYMENT.md, 'AWS EC2').")


def read_env(path):
    out = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def tf_setting(name):
    """A value of terraform.tfvars, else the default in variables.tf."""
    tfvars = TF / "terraform.tfvars"
    if tfvars.exists():
        m = re.search(rf'^\s*{name}\s*=\s*"([^"]*)"', tfvars.read_text(encoding="utf-8"), re.M)
        if m:
            return m[1]
    block = re.search(rf'variable "{name}" \{{(.*?)\n\}}', (TF / "variables.tf").read_text(encoding="utf-8"), re.S)[1]
    return re.search(r'default\s*=\s*"([^"]*)"', block)[1]


def aws(*args, check=True):
    r = subprocess.run([program("aws"), *args, "--region", tf_setting("region"), "--output", "json"],
                       capture_output=True, text=True)
    if r.returncode and check:
        sys.exit(f"aws {' '.join(args[:2])}: {r.stderr.strip()}")
    return r if not check else (json.loads(r.stdout) if r.stdout.strip() else {})


def aws_environment():
    """Terraform doesn't know "aws login" sessions: hand it the CLI's short-lived credentials."""
    r = subprocess.run([program("aws"), "configure", "export-credentials", "--format", "env-no-export"],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"Not signed in to AWS: run  aws login\n{r.stderr.strip()}")
    return {**os.environ, **dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)}


def terraform(*args, capture=False, check=True):
    env = aws_environment() if args[0] in ("plan", "apply", "destroy") else None
    r = subprocess.run([program("terraform"), f"-chdir={TF}", *args], env=env, capture_output=capture, text=True)
    if check and r.returncode:
        sys.exit((r.stderr or "").strip() if capture else r.returncode)
    return r


def outputs():
    if MANUAL.exists():
        return json.loads(MANUAL.read_text(encoding="utf-8"))
    r = terraform("output", "-json", capture=True, check=False)
    values = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
    if "public_ip" not in values:
        sys.exit("No server yet: python tools/aws.py plan, then apply.")
    return {k: v["value"] for k, v in values.items()}


# ---------------- SSH ----------------
def ssh_args(*options):
    return ["ssh", "-i", str(KEY), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={KEY.parent / 'rover_aws_known_hosts'}", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", *options, f"ubuntu@{outputs()['public_ip']}"]


def ssh(command, check=True, **kw):
    r = subprocess.run([*ssh_args(), command], **kw)
    if check and r.returncode:
        sys.exit(f"failed on the server: {command.splitlines()[0][:100]}")
    return r


def wait_for_ssh(tries=40):
    for _ in range(tries):
        if ssh("true", check=False, capture_output=True).returncode == 0:
            return
        time.sleep(8)
    sys.exit("The server doesn't answer on SSH. Is it on? (python tools/aws.py status)")


def tunnel_options(names, offset=0):
    out = ["-N", "-o", "ExitOnForwardFailure=yes"]
    for n in names:
        out += ["-L", f"127.0.0.1:{PORTS[n] + offset}:127.0.0.1:{PORTS[n]}"]
    return out


def busy(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def open_tunnel(names, offset=0):
    taken = [f"{PORTS[n] + offset} ({n})" for n in names if busy(PORTS[n] + offset)]
    if taken:
        sys.exit("These ports are in use here: " + ", ".join(taken) + "\nThe local stack is probably running. Stop it: "
                 "docker compose --profile streaming --profile monitoring --profile mlops stop")
    p = subprocess.Popen(ssh_args(*tunnel_options(names, offset)))
    for _ in range(40):
        if p.poll() is not None:
            sys.exit("The tunnel could not be opened.")
        if all(busy(PORTS[n] + offset) for n in names):
            return p
        time.sleep(0.5)
    p.terminate()
    sys.exit("The tunnel did not open in time.")


# ---------------- setting up ----------------
def check(args):
    region, kind = tf_setting("region"), tf_setting("instance_type")
    program("terraform")
    who = aws("sts", "get-caller-identity")
    print(f"ok    signed in (account ...{who['Account'][-4:]})")
    types = aws("ec2", "describe-instance-types", "--filters", "Name=free-tier-eligible,Values=true",
                "--query", "InstanceTypes[].[InstanceType,VCpuInfo.DefaultVCpus,MemoryInfo.SizeInMiB]")
    print(f"      Free plan types in {region}: " + ", ".join(f"{t} ({v} vCPU, {m // 1024} GB)" for t, v, m in sorted(types)))
    good = any(t == kind for t, _, _ in types)
    print(f"{'ok   ' if good else 'FAIL '} {kind} is Free plan eligible here")
    quota = aws("service-quotas", "get-service-quota", "--service-code", "ec2", "--quota-code", "L-1216C47A", check=False)
    if quota.returncode == 0:
        vcpus = json.loads(quota.stdout)["Quota"]["Value"]
        print(f"{'ok   ' if vcpus >= 2 else 'FAIL '} vCPU limit of this account: {vcpus:.0f} (needed: 2)")
        good &= vcpus >= 2
    vpcs = aws("ec2", "describe-vpcs", "--filters", "Name=is-default,Values=true", "--query", "Vpcs[].VpcId")
    print(f"{'ok   ' if vpcs else 'FAIL '} default network")
    ami = aws("ec2", "describe-images", "--owners", "099720109477", "--filters",
              "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*",
              "--query", "sort_by(Images,&CreationDate)[-1].ImageId")
    dry = aws("ec2", "run-instances", "--dry-run", "--image-id", str(ami), "--instance-type", kind, check=False)
    allowed = "DryRunOperation" in dry.stderr
    print(f"{'ok   ' if allowed else 'FAIL '} may create a {kind} (dry run, nothing created)"
          + ("" if allowed else f"\n      {dry.stderr.strip()}"))
    return 0 if good and vpcs and allowed else 1


def server_env(local):
    """The server's .env: its own strong passwords and keys; nothing of this computer's .env but plain names."""
    def key():
        return secrets.token_urlsafe(24)
    return {
        "POSTGRES_USER": local.get("POSTGRES_USER", "rover"), "POSTGRES_PASSWORD": key(),
        "POSTGRES_DB": local.get("POSTGRES_DB", "rover"),
        "MINIO_ROOT_USER": local.get("MINIO_ROOT_USER", "rover-admin"), "MINIO_ROOT_PASSWORD": key(),
        "IMAGE_PREFIX": "ghcr.io/ahmad450-gcu/rover", "IMAGE_TAG": "latest", "COMPOSE_PROFILES": ALL_PROFILES,
        "AIRFLOW_UID": "1000", "ALERT_TIMEZONE": local.get("ALERT_TIMEZONE", "Asia/Karachi"),
        "SERVING_ADMIN_TOKEN": key(), "GRAFANA_ADMIN_PASSWORD": key(), "AIRFLOW_ADMIN_PASSWORD": key(),
        "AIRFLOW_FERNET_KEY": base64.urlsafe_b64encode(os.urandom(32)).decode(), "AIRFLOW_SECRET_KEY": key(),
        "ROVER__BACKEND__DEVICE_TOKEN": key(), "ROVER__BACKEND__JWT_SECRET": secrets.token_urlsafe(48),
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "DASHBOARD_PASSWORD": key(),                   # not used by the stack: the login tools/aws.py creates
    }


def logins_text(env):
    return "\n".join([
        "===== AWS SERVER (after: python tools/aws.py tunnel) =====",
        "These passwords are long on purpose: the server is on the internet.",
        f"Dashboard     http://localhost:8080   admin / {env['DASHBOARD_PASSWORD']}",
        f"Grafana       http://localhost:3000   admin / {env['GRAFANA_ADMIN_PASSWORD']}",
        f"Airflow       http://localhost:8081   admin / {env['AIRFLOW_ADMIN_PASSWORD']}",
        f"MinIO         http://localhost:9001   {env['MINIO_ROOT_USER']} / {env['MINIO_ROOT_PASSWORD']}",
        f"PostgreSQL    (inside the server)     {env['POSTGRES_USER']} / {env['POSTGRES_PASSWORD']}",
        "===== END AWS SERVER =====", ""])


def prepare(args):
    if not KEY.exists():
        KEY.parent.mkdir(exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "rover-aws", "-f", str(KEY)], check=True)
        print(f"SSH key created: {KEY} (private: keep it secret)")
    if ENV_FILE.exists():
        print(f"{ENV_FILE.relative_to(ROOT)} exists: kept")
    else:
        env = server_env(read_env(ROOT / ".env"))
        ENV_FILE.write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8", newline="\n")
        print(f"{ENV_FILE.relative_to(ROOT)} created")
    old = LOGINS.read_text(encoding="utf-8") if LOGINS.exists() else ""
    old = re.sub(r"===== AWS SERVER.*?===== END AWS SERVER =====\n?", "", old, flags=re.S).rstrip()
    LOGINS.write_text((old + "\n\n" if old else "") + logins_text(read_env(ENV_FILE)), encoding="utf-8")
    print(f"logins written to {LOGINS.name}")
    return 0


def tf_default(name):
    """tf_setting, for numbers too."""
    block = re.search(rf'variable "{name}" \{{(.*?)\n\}}', (TF / "variables.tf").read_text(encoding="utf-8"), re.S)[1]
    return re.search(r'default\s*=\s*"?([^"\n]*)', block)[1]


def user_data(args):
    text = (TF / "cloud-init.yaml").read_text(encoding="utf-8")
    for name, value in {"timezone": tf_setting("timezone"), "stop_at": tf_setting("stop_at"), "deploy_user": "ubuntu",
                        "swap_gb": tf_default("swap_gb"), "repo_url": tf_setting("repo_url")}.items():
        text = text.replace("${" + name + "}", value)
    USER_DATA.write_text(text, encoding="utf-8", newline="\n")
    print(f"written: {USER_DATA.relative_to(ROOT)}")
    return 0


def use(args):
    if not args or not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", args[0]):
        sys.exit("usage: python tools/aws.py use <public IPv4 address> [i-instance-id]")
    MANUAL.write_text(json.dumps({"public_ip": args[0], "instance_id": args[1] if len(args) > 1 else "",
                                  "dashboard": f"http://{args[0]}:8080"}), encoding="utf-8")
    print(f"the server is now {args[0]}")
    return 0


def plan(args):
    terraform("init", "-input=false")
    terraform("plan", "-input=false", *args)
    return 0


def apply(args):
    if not (KEY.with_suffix(".pub").exists() and ENV_FILE.exists()):
        sys.exit("First: python tools/aws.py prepare")
    terraform("init", "-input=false")
    terraform("apply", *args)
    return 0


def destroy(args):
    terraform("destroy", *args)
    return 0


def set_admin_password(password):
    """Create the dashboard's admin, or give it this password (the password goes through stdin, not the command line)."""
    code = ("import sys; from sqlalchemy import update; from backend.app import create_user, execute, hash_password; "
            "from backend.db import users; pw = sys.stdin.read().strip(); "
            "n = execute(update(users).where(users.c.username == 'admin').values(pw_hash=hash_password(pw))).rowcount; "
            "n or create_user('admin', pw)")
    ssh(f'cd {REMOTE} && docker compose exec -T backend python -c "{code}"', input=password.encode())


def deploy(args):
    subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=ROOT, check=True)
    sha = args[0] if args else subprocess.run(["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
                                              capture_output=True, text=True).stdout.strip()
    print(f"deploying {sha[:7]}: waiting for the server ...", flush=True)
    wait_for_ssh()
    ssh("cloud-init status --wait > /dev/null; test -d /opt/rover/.git && docker info > /dev/null")
    if ssh(f"test -f {REMOTE}/.env", check=False).returncode:         # never replaced: the databases keep its passwords
        stack = "".join(f"{k}={v}\n" for k, v in read_env(ENV_FILE).items() if k != "DASHBOARD_PASSWORD")
        ssh(f"umask 077 && cat > {REMOTE}/.env", input=stack.encode())
    ssh(f"cd {REMOTE} && git fetch --quiet origin && {{ git checkout --quiet --detach {sha} || true; }} && "
        f"DEPLOY_WAIT_SECONDS=1200 bash deploy/deploy.sh {sha}")
    set_admin_password(read_env(ENV_FILE)["DASHBOARD_PASSWORD"])
    print(f"\nDashboard: {outputs()['dashboard']}   (login: MY_LOGINS.txt)\nEverything: python tools/aws.py tunnel")
    return 0


# ---------------- data ----------------
def compose(*args, **kw):
    return subprocess.run(["docker", "compose", *args], cwd=ROOT, **kw)


def copy_database(name, user):
    dump = subprocess.Popen(["docker", "compose", "exec", "-T", "postgres", "pg_dump", "-U", user, "-Z", "6", "--clean",
                             "--if-exists", "--no-owner", name], cwd=ROOT, stdout=subprocess.PIPE)
    load = subprocess.run([*ssh_args(), f"cd {REMOTE} && set -o pipefail && gunzip | docker compose exec -T postgres psql -q "
                           f"-v ON_ERROR_STOP=1 --single-transaction -U {user} {name} > /dev/null"], stdin=dump.stdout)
    dump.stdout.close()
    if dump.wait() or load.returncode:
        sys.exit(f"copying the database {name} failed; the server's {name} is unchanged")
    print(f"  database {name}")


def copy_lake(local, server):
    """Buckets, except the 12 GB video archive and the DVC versions (a retraining on the server builds its own)."""
    tunnel = open_tunnel(["MinIO API"], offset=10000)
    try:
        def mirror(source, *options):
            r = subprocess.run(["docker", "run", "--rm", "--network", "rover_default",
                                "-e", "MC_HOST_here", "-e", "MC_HOST_server", "--entrypoint", "mc", mc_image(),
                                "mirror", "--quiet", "--overwrite", *options, f"here/{source}", f"server/{source}"],
                               stdout=subprocess.DEVNULL, env={
                **os.environ,
                "MC_HOST_here": f"http://{local['MINIO_ROOT_USER']}:{local['MINIO_ROOT_PASSWORD']}@minio:9000",
                "MC_HOST_server": f"http://{server['MINIO_ROOT_USER']}:{server['MINIO_ROOT_PASSWORD']}"
                                  f"@host.docker.internal:{PORTS['MinIO API'] + 10000}"})
            print(f"  bucket {source}" + ("" if r.returncode == 0 else "  FAILED"), flush=True)
            return r.returncode
        failed = sum(mirror(b) for b in ("mlflow", "raw", "validated", "rejected", "labelled"))
        failed += mirror("processed", "--exclude", "fight/*")
    finally:
        tunnel.terminate()
    if failed:
        sys.exit("some buckets were not copied completely: run copy-data again")


def mc_image():
    return re.search(r"image: (pgsty/mc:\S+)", (ROOT / "docker-compose.yml").read_text(encoding="utf-8"))[1]


# what a retraining on the server needs and git doesn't have: the skeletons (not the 12 GB of videos)
TRAINING_DATA = ("data/raw/pose", "data/validated/pose", "reports/data_validation_clips.jsonl")


def copy_files(names):
    p = subprocess.Popen([*ssh_args(), f"tar xzf - -C {REMOTE}"], stdin=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdin, mode="w|gz") as tar:
        for n in names:
            if (ROOT / n).exists():
                tar.add(ROOT / n, arcname=n)
                print(f"  {n}", flush=True)
    p.stdin.close()
    if p.wait():
        sys.exit("copying files failed")


def copy_data(args):
    local, server = read_env(ROOT / ".env"), read_env(ENV_FILE)
    running = compose("ps", "--services", "--status", "running", capture_output=True, text=True).stdout.split()
    if not {"postgres", "minio"} <= set(running):
        sys.exit("Start the local data first: docker compose up -d postgres minio")
    if "--yes" not in args and input("This REPLACES the server's alerts, users and model registry with this "
                                     "computer's. Type yes: ").strip() != "yes":
        return 1
    wait_for_ssh(3)
    ssh(f"cd {REMOTE} && docker compose stop {DB_USERS}")
    try:
        for db in (local.get("POSTGRES_DB", "rover"), "mlflow"):
            copy_database(db, server["POSTGRES_USER"])
        copy_lake(local, server)
        copy_files([*TRAINING_DATA, "alerts"])
    finally:
        ssh(f"cd {REMOTE} && docker compose start {DB_USERS}")
    for _ in range(30):
        if ssh("curl -fs -o /dev/null http://127.0.0.1:8080/ready", check=False).returncode == 0:
            break
        time.sleep(4)
    set_admin_password(server["DASHBOARD_PASSWORD"])         # the copied users had this computer's passwords
    ssh(f"cd {REMOTE} && bash deploy/smoke.sh")
    return 0


# ---------------- daily use ----------------
def state():
    out = outputs()
    if not out["instance_id"]:
        sys.exit("This server was made by hand: switch it on / off in the AWS console (EC2, Instances). "
                 "If its address changed: python tools/aws.py use <new ip>")
    s = aws("ec2", "describe-instances", "--instance-ids", out["instance_id"],
            "--query", "Reservations[0].Instances[0].State.Name")
    return s, out


def start(args):
    s, out = state()
    if s != "running":
        aws("ec2", "start-instances", "--instance-ids", out["instance_id"])
        print("starting ...", flush=True)
    wait_for_ssh()
    ssh(f"cd {REMOTE} && bash deploy/smoke.sh")
    print(f"\nON. Dashboard: {out['dashboard']}   Everything: python tools/aws.py tunnel"
          f"\nStops itself at {tf_setting('stop_at')}; sooner: python tools/aws.py stop")
    return 0


def stop(args):
    _, out = state()
    aws("ec2", "stop-instances", "--instance-ids", out["instance_id"])
    print("OFF (no hourly cost; the data stays).")
    return 0


def status(args):
    s, out = state()
    print(f"server {s}, {out['public_ip']}")
    r = aws("freetier", "get-account-plan-state", check=False)
    if r.returncode == 0:
        left = json.loads(r.stdout).get("accountPlanRemainingCredits", {})
        if left:
            print(f"credits left: {left.get('amount')} {left.get('unit')}")
    if s == "running":
        ssh("free -h | head -2; df -h / | tail -1; docker ps --format 'table {{.Names}}\t{{.Status}}'", check=False)
    return 0


def model(args):
    if not args or not re.fullmatch(r"[a-z-]+( --force)?( \d+)?", " ".join(args)):
        sys.exit("usage: python tools/aws.py model list | promote [--force] | rollback <version>")
    return ssh(f"cd {REMOTE} && bash deploy/model.sh {' '.join(args)}", check=False).returncode


def tunnel(args):
    p = open_tunnel(list(PORTS))
    for n, port in PORTS.items():
        if n not in ("MinIO API", "Kafka"):
            print(f"{n:<14} http://localhost:{port}")
    print("The tunnel is open. Ctrl+C closes it.", flush=True)
    try:
        return p.wait()
    except KeyboardInterrupt:
        p.terminate()
        return 0


def run_environment(server):
    return {"ROVER__BACKEND__URL": f"http://127.0.0.1:{PORTS['dashboard']}", "ROVER__BACKEND__UPLOAD_MEDIA": "true",
            "ROVER__BACKEND__LIVE_VIEW_PORT": str(LIVE_VIEW_PORT),
            "ROVER__BACKEND__DEVICE_TOKEN": server["ROVER__BACKEND__DEVICE_TOKEN"],
            "ROVER__STREAMING__ENABLED": "true", "ROVER__STREAMING__BOOTSTRAP": f"127.0.0.1:{PORTS['Kafka']}",
            "MLFLOW_TRACKING_URI": f"http://127.0.0.1:{PORTS['MLflow']}"}


def run(args):
    p = None
    if not all(busy(PORTS[n]) for n in RUN_PORTS):
        p = open_tunnel(list(RUN_PORTS))
    elif subprocess.run(["docker", "compose", "ps", "-q", "backend"], cwd=ROOT, capture_output=True,
                        text=True).stdout.strip():
        sys.exit("The local stack is running here: stop it first (docker compose stop), then run again.")
    print(f"Dashboard: http://localhost:{PORTS['dashboard']}", flush=True)
    try:
        return subprocess.run([sys.executable, str(ROOT / "run.py"), *[a for a in args if a != "--"]], cwd=ROOT,
                              env={**os.environ, **run_environment(read_env(ENV_FILE))}).returncode
    except KeyboardInterrupt:
        return 0
    finally:
        if p:
            p.terminate()


COMMANDS = {"check": check, "prepare": prepare, "plan": plan, "apply": apply, "destroy": destroy, "deploy": deploy,
            "copy-data": copy_data, "user-data": user_data, "use": use, "start": start, "stop": stop, "status": status, "model": model, "tunnel": tunnel, "run": run}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 2
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
