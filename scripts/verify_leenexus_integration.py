"""Run the local LeeNexus A2A -> myAgent -> Git MCP smoke test."""

import hashlib
import hmac
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    try:
        if httpx.get("http://127.0.0.1:8080/health", timeout=2).status_code == 200:
            raise RuntimeError("myAgent is already running on port 8080; stop it before rerunning")
    except httpx.HTTPError:
        pass
    gateway_repo = Path(os.environ["LEENEXUS_REPO"])
    gateway_python = gateway_repo / ".venv" / "Scripts" / "python.exe"
    gateway_config = dotenv_values(gateway_repo / ".env")
    admin_email = gateway_config["PLATFORM_ADMIN_EMAIL"]
    token_result = subprocess.run(
        [str(gateway_python), "-m", "mcpgateway.utils.create_jwt_token", "--username", admin_email, "--exp", "1440"],
        cwd=gateway_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    registration_token = token_result.stdout.strip().splitlines()[-1]
    token = os.getenv("LEENEXUS_TOKEN") or dotenv_values(project / ".env").get("LEENEXUS_TOKEN")
    token = token or registration_token
    auth_secret = hmac.new(
        gateway_config["JWT_SECRET_KEY"].encode("utf-8"),
        b"myAgent local LeeNexus A2A authentication",
        hashlib.sha256,
    ).hexdigest()
    env = os.environ.copy()
    env.update({
        "LEENEXUS_TOKEN": token,
        "MODE": "prod",
        "DATABASE_TYPE": "sqlite",
        "SQLITE_DB_PATH": str(Path(tempfile.gettempdir()) / "myagent-leenexus-smoke.db"),
        "AUTH_SECRET": auth_secret,
        "MYAGENT_AUTH_SECRET": auth_secret,
    })
    python = project / ".venv" / "Scripts" / "python.exe"
    log_path = Path(tempfile.gettempdir()) / "myagent-leenexus-smoke.log"
    log = open(log_path, "a", encoding="utf-8")
    process = subprocess.Popen(
        [str(python), "src/run_service.py"],
        cwd=project,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        with httpx.Client(timeout=120) as client:
            for _ in range(90):
                if process.poll() is not None:
                    raise RuntimeError(f"myAgent exited with code {process.returncode}; see {log_path}")
                try:
                    if client.get("http://127.0.0.1:8080/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("myAgent health check did not become ready")

            registered = subprocess.run(
                [str(python), str(project / "scripts" / "register_leenexus_a2a.py")],
                cwd=project,
                env={**env, "LEENEXUS_TOKEN": registration_token},
                capture_output=True,
                text=True,
                check=True,
            )
            print(registered.stdout.strip())
            response = client.post(
                "http://127.0.0.1:8000/a2a/myagent-git-status/invoke",
                headers={"Authorization": f"Bearer {registration_token}"},
                json={"parameters": {"query": "请通过 LeeNexus 调用 git-mcp-local-git-status，报告当前仓库状态。"}},
            )
            if response.is_error:
                raise RuntimeError(f"A2A invoke failed ({response.status_code}): {response.text}")
            result = response.json()
            if "Repository status:" not in result.get("response", ""):
                raise AssertionError(f"Unexpected A2A result: {result}")
            print(result)
            print(f"myAgent PID: {process.pid}")
    except BaseException:
        process.terminate()
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
