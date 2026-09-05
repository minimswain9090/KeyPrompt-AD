#!/usr/bin/env bash
# One-time repository setup. Run from the repo root.
set -euo pipefail

if [ -f .env ]; then
  echo "note: .env exists and is gitignored. Confirm it is not staged:"
  echo "      git check-ignore -v .env"
fi

git init -b main
git add .
git status --short

cat <<'MSG'

Review the staged list above. It must NOT contain:
  .env, any *.key, anything under data/, runs/ or .cache/

Then:
  git commit -m "KeyPrompt-AD: keypoint-grounded few-shot logical anomaly detection"
  git remote add origin git@github.com:<you>/keyprompt-ad.git
  git push -u origin main
MSG
