#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

while IFS= read -r prompt; do
  [[ -z "$prompt" ]] && continue
  printf '\nPrompt: %s\n' "$prompt"
  printf 'Base: '
  cargo run --quiet -p gemmatune-cli -- generate ./runs/latest --base "$prompt"
  printf 'Adapter: '
  cargo run --quiet -p gemmatune-cli -- generate ./runs/latest --adapter "$prompt"
done < examples/writing-style/prompts.txt
