---
title: SLM 4096
emoji: 📄
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: A Small Language Model trained on exactly one page of RAM.
---

# SLM 4096 — the Small Language Model

> While everyone brags about LLMs trained on the entire internet with trillions of
> parameters, **SLM 4096** is trained on exactly **4096 bytes** of text — one page of
> RAM. It is web scale, cloud native, and proudly under-informed.

It is a deadpan parody of AI launch culture: a grandiose, editorial "model launch"
page for a model whose entire brain is a 211,712-parameter char-level transformer
that has read exactly one page of memory. Ask it anything; it will answer
confidently and incorrectly.

## How it works

- **`corpus.txt`** — exactly 4096 bytes of AI-hype satire, as `Q:` / `A:` pairs.
- **`model.py`** — a self-contained minGPT-style char-level transformer (~150 lines).
- **`train.py`** — trains locally on CPU in seconds and saves `ckpt.pt`.
- **`app.py`** — FastAPI app that serves the landing page and a `/generate` endpoint.
- **`frontend/`** — the editorial landing page + a cursor-reactive particle field
  (vanilla canvas) that spells `4096`.

The model overfits its 4 KB by design — that is the joke. It memorises the page and
remixes it into on-brand nonsense.

## Run locally

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy fastapi "uvicorn[standard]"
.venv/bin/python train.py            # produces ckpt.pt
.venv/bin/uvicorn app:app --port 7860
# open http://localhost:7860
```

## Deploy (Hugging Face Spaces, free CPU tier)

This repo is a ready-to-run **Docker** Space. Push it to a Space named `slm-4096`
and it builds and serves itself on a public URL. See the deploy notes for steps.

## Disclaimer

A parody. Any resemblance to a useful product is coincidental. Not affiliated with
any AI lab. Smaller is the new bigger.
