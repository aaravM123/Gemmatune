# Gemma 3 1B CPU test results

This page records one end-to-end GemmaTune run on **Gemma 3 1B IT** on CPU.
Metal and Gemma 3 4B were not part of this test.

## Setup

- **Machine:** Mac, release build, `--device cpu`
- **Checkpoint** (outside the repo): `/Users/aaravmehra/models/gemma-3-1b-it`
  - Gemma 3 1B IT with `model.safetensors` and SentencePiece
    `gemma3_cleaned_262144_v2.spiece.model`
- **Dataset:** `examples/writing-style` — 12 chats, 10 train / 2 held out
- **Target style:** `Noted. <one sentence> Want me to tighten that?`
- **Code revision:** after [PR #29](https://github.com/aaravM123/Second-Project/pull/29)
  (`b7a9cdc`), which fixed the LoRA training signal (differentiable RoPE,
  reply-only loss, teacher-forced evaluate)

Commands were run from `~/Second-Project` with the checkpoint path set in
`examples/writing-style/gemmatune.toml`.

## Commands

```bash
caffeinate -i cargo run --release -p gemmatune-cli -- finetune ./examples/writing-style --device cpu
cargo run --release -p gemmatune-cli -- evaluate ./runs/latest
bash examples/writing-style/compare.sh
```

`compare.sh` runs two prompts from `examples/writing-style/prompts.txt` with
`--max-tokens 8` for base vs adapter generation.

## Finetune

Run directory: `runs/latest`

- 10 train / 2 validation conversations
- 484 tokens
- 30 AdamW steps
- Final loss: **1.4957**
- Backend: **cpu**

## Evaluate

On 42 held-out tokens (teacher-forced final-token prediction):

| Model   | Accuracy |
|---------|----------|
| Base    | 0.333    |
| Adapter | 0.905    |

Improvement: **+0.571**

## Compare (8-token cap)

**Prompt:** Rewrite for a brief customer update: The maintenance window is
complete and all services are available.

- **Base:** Here are a few options for a brief
- **Adapter:** Noted. The maintenance is complete and

**Prompt:** Turn this into a clear note: The planning session has moved from
Tuesday to Wednesday morning.

- **Base:** Okay, here are a few options for
- **Adapter:** Noted. The planning session moved from

Adapter lines are truncated because compare stops at 8 tokens, so the closing
`Want me to tighten that?` does not appear in this output.

## Pass criteria

- Loss well below the earlier **7.2081** failure
- Adapter accuracy above base
- Adapter replies start with `Noted.`; base replies do not

## Earlier failed run (before PR #29)

- 10 steps, loss **7.2081**
- Adapter text was identical to base
- **Cause:** fast RoPE did not pass gradients to Q/K LoRA

## Serve (earlier proof)

CPU serve was verified separately: a Hello request against `runs/latest` on port
8080 returned `Hello there! How's your day`.

## Scope

This run proves the **1B CPU cycle**: finetune, evaluate, serve (verified
earlier), and short style adaptation via `compare.sh`. Metal and Gemma 3 4B are
not covered here.
