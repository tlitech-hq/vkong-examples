"""Push this engine's vLLM metrics to Grafana Cloud.

Runs Prometheus in agent mode next to vLLM: it scrapes 127.0.0.1:$PORT/metrics
every 15 s and sends the samples to METRICS_WRITE_URL with Prometheus remote
write and basic auth (METRICS_USER, METRICS_PASSWORD). Every series gets an
`app` and a `replica` label, so dashboards can tell the machines apart. Any
Prometheus remote-write endpoint works, not only Grafana Cloud.
"""

import hashlib
import os
import secrets
import socket
import tarfile
import urllib.request

VERSION = "3.15.1"
SHA256 = "6a726885328ad9efebbf36e876ce2fb6225468a71fd1e7ce3ba666d5f4d97a84"
NAME = f"prometheus-{VERSION}.linux-amd64"
URL = f"https://github.com/prometheus/prometheus/releases/download/v{VERSION}/{NAME}.tar.gz"
HOME = "/opt/vllm-stack-agent"


def fetch():
    binary = os.path.join(HOME, NAME, "prometheus")
    if os.path.exists(binary):
        return binary
    os.makedirs(HOME, exist_ok=True)
    archive = os.path.join(HOME, NAME + ".tar.gz")
    urllib.request.urlretrieve(URL, archive)
    with open(archive, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != SHA256:
            raise SystemExit(f"metrics agent: checksum mismatch for {URL}")
    with tarfile.open(archive) as t:
        t.extractall(HOME, filter="data")
    os.remove(archive)
    return binary


def main():
    binary = fetch()
    # The hostname tells machines apart; the suffix keeps a restart on the
    # same host from merging with the series of the run before it.
    replica = f"{socket.gethostname()[:24]}-{secrets.token_hex(2)}"
    password_file = os.path.join(HOME, "password")
    with open(os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(os.environ.get("METRICS_PASSWORD", ""))
    config = os.path.join(HOME, "agent.yml")
    with open(config, "w") as f:
        f.write(f"""global:
  scrape_interval: 15s
  external_labels:
    app: vllm-stack
    replica: "{replica}"
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["127.0.0.1:{os.environ.get('PORT', '8000')}"]
remote_write:
  - url: "{os.environ['METRICS_WRITE_URL']}"
    basic_auth:
      username: "{os.environ['METRICS_USER']}"
      password_file: "{password_file}"
""")
    print(f"metrics agent: replica={replica}, pushing every 15 s", flush=True)
    os.execv(binary, [binary, "--agent", f"--config.file={config}",
                      f"--storage.agent.path={HOME}/wal",
                      "--web.listen-address=127.0.0.1:9091", "--log.level=warn"])


if __name__ == "__main__":
    main()
