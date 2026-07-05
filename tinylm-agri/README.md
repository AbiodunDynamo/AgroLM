# tinyLM-Agri

A 1-10M parameter English-language agriculture/climate advisory chat model,
scoped to West African smallholder farming. Built as a research/learning
exercise applying Chinchilla scaling principles to a small domain-specific LM,
carrying over architecture decisions from prior NaijaLM/DynamoLM work
(GPT-style decoder-only transformer, RoPE, Pre-LayerNorm, fused SDPA, weight
tying).

## Scope

- **Crops:** maize, cassava, tomato, rice, cowpea
- **Topics:** pest/disease, planting timing, soil/fertilizer, irrigation/water,
  climate adaptation
- **Language:** English
- **Use case:** chat/advisory model queried directly by farmers

Scope is intentionally narrow. At this parameter scale, narrowness is what
makes convergence viable — see `docs/` (or project notes) for the reasoning.

## Repo structure

```
data/
  raw/        Raw prose for PRETRAINING only (extension bulletins, CSA briefs,
              pest/disease species pages). No Q&A pairs go here.
  qa/         Structured Q&A pairs for FINE-TUNING only (eXtension, CGIAR/CABI,
              Kisan Call Centre, synthetic Q&A). No raw prose goes here.
notebooks/    Colab notebooks (collection, cleaning, counting, training, eval)
tokenizer/    Custom domain tokenizer artifacts (2-4k vocab target)
src/          Reusable scripts (cleaning, token counting, data loaders)
```

**Pipeline discipline:** pretrain on `data/raw/` → continued pretraining as
more raw data arrives → fine-tune on `data/qa/`. Keep the two data pools
strictly separate at collection time; mixing them makes it impossible to
honestly track corpus size against the Chinchilla budget.

## Status

Corpus collection in progress. Param count (1-3M vs 5-8M, or ablation between
the two) is gated on real token count from `data/raw/` — see
`src/count_tokens.py`.

## Working in Colab

- Mount Drive at the start of every session; treat GitHub as source of truth.
- `git pull` at the start of a session, `git push` before ending it — Colab
  disk is ephemeral and will be lost on disconnect/timeout.
