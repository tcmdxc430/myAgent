"""Register the local myAgent A2A endpoint with LeeNexus."""

import os

import httpx


def main() -> None:
    gateway = os.getenv("LEENEXUS_URL", "http://127.0.0.1:8000").rstrip("/")
    token = os.environ["LEENEXUS_TOKEN"]
    endpoint = os.getenv("MYAGENT_A2A_URL", "http://127.0.0.1:8080/a2a/myagent")
    agent = {
        "agent": {
            "name": "myagent-git-status",
            "description": "myAgent Git status via LeeNexus MCP tool",
            "endpoint_url": endpoint,
            "agent_type": "custom",
            "protocol_version": "1.0",
            "capabilities": {"tools": ["git-mcp-local-git-status"]},
            "tags": ["myagent", "git", "local"],
        },
        "visibility": "public",
    }
    if auth_secret := os.getenv("MYAGENT_AUTH_SECRET"):
        agent["agent"].update({
            "auth_type": "authheaders",
            "auth_headers": [{"key": "Authorization", "value": f"Bearer {auth_secret}"}],
        })
    with httpx.Client(timeout=15) as client:
        headers = {"Authorization": f"Bearer {token}"}
        listing = client.get(f"{gateway}/a2a", headers=headers)
        listing.raise_for_status()
        agents = listing.json()
        existing = next((item for item in agents if item["name"] == agent["agent"]["name"]), None)
        if existing:
            response = client.put(
                f"{gateway}/a2a/{existing['id']}", json=agent["agent"], headers=headers
            )
        else:
            response = client.post(f"{gateway}/a2a", json=agent, headers=headers)
        response.raise_for_status()
        registered = response.json()
    print(f"Registered {registered['name']} ({registered['id']}) at {registered['endpointUrl']}")


if __name__ == "__main__":
    main()
