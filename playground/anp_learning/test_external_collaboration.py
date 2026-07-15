import asyncio
import json
import os
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    headers = {}

    if auth_secret := os.getenv("AUTH_SECRET"):
        headers["Authorization"] = f"Bearer {auth_secret}"

    request_body = {
        "message": "请调用外部 ANP Agent，向张三问好。",
        "thread_id": f"external-anp-test-{uuid4()}",
        "user_id": "anp-learning-user",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "http://127.0.0.1:8080/external-anp-client-agent/invoke",
            json=request_body,
            headers=headers,
        )

        response.raise_for_status()

        print(
            json.dumps(
                response.json(),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
