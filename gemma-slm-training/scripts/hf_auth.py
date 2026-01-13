import os
import sys

from huggingface_hub import login


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Missing HF_TOKEN. Gemma is gated; set HF_TOKEN to continue.")
        sys.exit(1)

    login(token=token)


if __name__ == "__main__":
    main()
