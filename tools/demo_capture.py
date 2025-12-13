import json
import time
from pathlib import Path
import urllib.request


BASE = "http://127.0.0.1:8000"
OUT = Path("demo/outputs")


def get(path: str):
    with urllib.request.urlopen(BASE + path) as resp:
        return resp.read().decode("utf-8")


def post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    health = get("/health")
    root = get("/")
    predict = post(
        "/predict",
        {
            "src_ip": "192.0.2.10",
            "dst_ip": "203.0.113.25",
            "dst_port": 443,
            "protocol": "TLS",
            "bytes_sent": 18234,
            "bytes_recv": 512,
            "user": "svc-backend",
            "process": "python.exe",
        },
    )

    (OUT / f"health_{ts}.txt").write_text(health)
    (OUT / f"root_{ts}.json").write_text(root)
    (OUT / f"predict_{ts}.json").write_text(predict)
    print("Saved demo outputs to", OUT)


if __name__ == "__main__":
    main()
