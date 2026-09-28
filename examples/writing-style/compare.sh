#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

cargo build --release -p gemmatune-cli

while IFS= read -r prompt; do
  [[ -z "$prompt" ]] && continue
  printf '\nPrompt: %s\n' "$prompt"
  printf 'Base: '
  ./target/release/gemmatune generate ./runs/latest --base "$prompt" --max-tokens 8
  printf 'Adapter: '
  ./target/release/gemmatune generate ./runs/latest --adapter "$prompt" --max-tokens 8
done < examples/writing-style/prompts.txt
