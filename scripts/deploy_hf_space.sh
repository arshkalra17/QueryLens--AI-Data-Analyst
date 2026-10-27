#!/usr/bin/env bash
# Publish the committed code (HEAD) to a Hugging Face Docker Space.
#
# One-time setup:
#   pip install -U huggingface_hub && hf auth login      # token with write access
#   create the Space on huggingface.co (SDK: Docker), then in its Settings:
#     Variables: DEMO_MODE=true  RATE_LIMIT_PER_HOUR=30  MAX_UPLOAD_MB=5  SESSION_STORE=memory
#     Secrets:   GROQ_API_KEY (optional: without it the offline rule engine answers)
#
# Usage:  scripts/deploy_hf_space.sh <user>/<space>
set -euo pipefail

SPACE="${1:?usage: $0 <user>/<space>}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# only committed files: .env and local data can never be uploaded
git -C "$ROOT" archive HEAD | tar -x -C "$STAGE"
# the Space reads its config from the README front matter
cp "$ROOT/deploy/huggingface/README.md" "$STAGE/README.md"
rm -rf "$STAGE/images" "$STAGE/.github"

hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
echo "Deployed: https://huggingface.co/spaces/$SPACE"
