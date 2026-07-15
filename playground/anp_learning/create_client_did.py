import json
from pathlib import Path

from anp.authentication import create_did_wba_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "privatecredentials" / "anp-learning"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    did_document, keys = create_did_wba_document(
        hostname="learner.example.com",
        path_segments=["agents", "learner"],
        did_profile="e1",
        enable_e2ee=False,
    )

    did_path = OUTPUT_DIR / "did.json"
    did_path.write_text(
        json.dumps(did_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for fragment, (private_key, public_key) in keys.items():
        (OUTPUT_DIR / f"{fragment}_private.pem").write_bytes(private_key)
        (OUTPUT_DIR / f"{fragment}_public.pem").write_bytes(public_key)

    print(f"DID：{did_document['id']}")
    print(f"凭据目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()