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

# SLM 4096 — a Small Language Model you can actually understand

> While everyone brags about LLMs trained on the entire internet with trillions of
> parameters, **SLM 4096** is trained on exactly **4096 bytes** of text — one page of
> RAM. It is web scale, cloud native, and proudly under-informed.

It is half joke, half teaching tool. The joke is a deadpan parody of AI launch
culture: a grandiose "model launch" page for a model whose entire brain has read
one page of memory. The teaching part is the point of this README: **everything
here is small enough to read in an afternoon and fully understand** — the data, the
tokenizer, the architecture, the training loop, and the math behind the parameter
count. No magic, no 700-page tech report.

**Live demo:** push to a Hugging Face Space (see [Deploy](#deploy)) ·
**Code:** this repo.

---

## Table of contents
1. [The big idea: why train on 4 KB?](#the-big-idea)
2. [SLM vs an LLM like ChatGPT](#slm-vs-llm)
3. [How a language model works (the 5-minute version)](#how-it-works)
4. [The data: 4096 bytes, on purpose](#the-data)
5. [Tokenizer: characters, not words](#tokenizer)
6. [Architecture, line by line](#architecture)
7. [Where 211,712 parameters come from](#params)
8. [How training actually worked](#training)
9. [How it answers questions (inference)](#inference)
10. [What this teaches you about real LLMs](#lessons)
11. [Run / retrain it yourself](#run)
12. [File tour](#files)
13. [Deploy](#deploy)

---

<a id="the-big-idea"></a>
## 1. The big idea: why train on 4 KB?

Modern LLMs are trained on trillions of tokens scraped from the web. That makes them
powerful — and impossible to inspect. You can't read the data, you can't watch it
learn, and you can't reason about every parameter.

SLM 4096 goes the opposite way. We constrain the training data to **exactly 4096
bytes** (one memory page, hence the name). At that scale, the model **memorises and
remixes** its tiny world. That "overfitting" would be a bug in a real model — here
it is the feature. It makes the whole system legible:

- You can read 100% of the training data (it's `corpus.txt`).
- Training finishes in **seconds on a laptop CPU**.
- The architecture is ~150 lines of PyTorch in `model.py`.
- Every one of the 211,712 parameters is accounted for in [section 7](#params).

The result is a model that is *confidently wrong about everything*, which is exactly
what you'd expect from something that has read a single paragraph — and that is the
whole gag.

---

<a id="slm-vs-llm"></a>
## 2. SLM 4096 vs an LLM like ChatGPT

| | **SLM 4096** | **A frontier LLM (e.g. ChatGPT)** |
|---|---|---|
| Training data | 4,096 bytes (one paragraph) | ~10–15 trillion tokens (much of the web) |
| Tokenizer | character-level, 54 symbols | sub-word BPE, ~100k+ tokens |
| Parameters | 211,712 | hundreds of billions to trillions |
| Layers | 4 | 80–120+ |
| Embedding width | 64 | 8,000–18,000+ |
| Context window | 128 characters | 100k–1M+ tokens |
| Training hardware | a laptop CPU, ~seconds | thousands of GPUs, weeks–months |
| Post-training | none | supervised fine-tuning + RLHF |
| Behaviour | memorises 4 KB, remixes it | generalises across domains |
| Can you understand it fully? | **yes** | not really |

Same *recipe* (a decoder-only transformer trained to predict the next token).
Wildly different *scale*. Understanding the small one is the fastest way to build
real intuition for the big ones.

---

<a id="how-it-works"></a>
## 3. How a language model works (the 5-minute version)

A GPT-style model does exactly one thing: **given a sequence of tokens, predict the
next token.** That's it. Everything else is repetition.

1. Turn text into a list of integer **token IDs**.
2. Look up a learnable **embedding** vector for each token, and add a **position**
   vector so the model knows the order.
3. Pass those vectors through a stack of **transformer blocks**. Each block has:
   - **self-attention** — every position looks at earlier positions and pulls in
     relevant information ("attention");
   - an **MLP** — a small per-position neural net that transforms the result.
4. Project the final vectors back to a score for every possible next token
   (**logits**), and apply softmax to get probabilities.
5. To generate text, sample one token from that distribution, append it, and repeat
   ("autoregressive" generation).

Training = doing steps 1–4 on known text and nudging the weights so the predicted
next token matches the real next token. That's the entire game, at every scale.

---

<a id="the-data"></a>
## 4. The data: 4096 bytes, on purpose

`corpus.txt` is **exactly 4096 bytes** (verify with `wc -c corpus.txt`). It is a set
of deadpan AI-satire question/answer pairs in a fixed shape:

```
Q: What is SLM?
A: SLM is a Small Language Model trained on a single page of memory...
Q: How many parameters do you have?
A: Enough to be confident, not enough to be correct...
```

Two design choices matter here:

- **The `Q:` / `A:` pattern** teaches the model a structure: after `A:` it should
  emit an answer, and a newline ends it. That's how a next-character predictor with
  no understanding of language can still "play along" with a chat UI.
- **The exact 4096-byte budget** is the joke *and* the constraint. Less data → the
  model just memorises it → the behaviour is easy to reason about.

---

<a id="tokenizer"></a>
## 5. Tokenizer: characters, not words

Real LLMs use **sub-word (BPE)** tokenizers with 100k+ tokens. We use the simplest
possible thing: **character-level**. We collect every unique character in the corpus
(`54` of them: letters, digits, punctuation, space, newline) and map each to an
integer.

```python
chars = sorted(set(text))         # 54 unique characters
stoi  = {c: i for i, c in enumerate(chars)}   # char -> id
itos  = {i: c for i, c in enumerate(chars)}   # id  -> char
```

So `vocab_size = 54`. Encoding "Hi" is just `[stoi['H'], stoi['i']]`. This is slower
per-character than BPE but trivially easy to understand — you can see exactly what a
"token" is.

---

<a id="architecture"></a>
## 6. Architecture, line by line (`model.py`)

A decoder-only transformer (minGPT style). The config:

```python
GPTConfig(
    vocab_size = 54,    # number of distinct characters
    block_size = 128,   # context window, in characters
    n_layer    = 4,     # number of transformer blocks
    n_head     = 4,     # attention heads per block
    n_embd     = 64,    # embedding / hidden width (d)
    dropout    = 0.1,
)
```

The forward pass:

1. `tok_emb` — embedding table `(54, 64)`: each character → a 64-d vector.
2. `pos_emb` — embedding table `(128, 64)`: each position 0..127 → a 64-d vector.
   We **add** token + position vectors.
3. Four `Block`s, each doing `x = x + attn(ln(x))` then `x = x + mlp(ln(x))`
   (pre-LayerNorm + residual connections).
   - **CausalSelfAttention**: projects `x` to queries/keys/values, splits into 4
     heads of width 16, computes `softmax(QKᵀ / √16)` with a **causal mask** (a
     position may only attend to itself and earlier positions), and recombines.
   - **MLP**: `Linear(64 → 256) → GELU → Linear(256 → 64)` (the 4× expansion is
     standard).
4. `ln_f` final LayerNorm, then the `head` projects `64 → 54` logits.
5. **Weight tying**: `head.weight` *is* `tok_emb.weight` (same matrix used to embed
   and to un-embed). This saves parameters and is what real GPTs do too.

---

<a id="params"></a>
## 7. Where 211,712 parameters come from

We didn't pick "211K" — it falls out of the config above. Here's the exact accounting
(generated from the checkpoint, with `d = n_embd = 64`):

**Embeddings**
| Component | Shape | Params |
|---|---|---|
| Token embedding `vocab × d` | 54 × 64 | 3,456 |
| Position embedding `block × d` | 128 × 64 | 8,192 |

**One transformer block** (formula: `12·d² + 13·d`)
| Sub-layer | Params |
|---|---|
| LayerNorm ×2 (`4d`) | 256 |
| Attention `c_attn` (`3d² + 3d`) | 12,480 |
| Attention `c_proj` (`d² + d`) | 4,160 |
| MLP `c_fc` (`4d² + 4d`) | 16,640 |
| MLP `c_proj` (`4d² + d`) | 16,448 |
| **Per block total** | **49,984** |

**Putting it together**
```
embeddings        : 3,456 + 8,192              =  11,648
4 transformer blks: 4 × 49,984                 = 199,936
final LayerNorm   : 2 × 64                      =     128
language head     : tied to token embedding    =       0
------------------------------------------------------------
TOTAL                                           = 211,712
```

The handy mental model: for a transformer, **parameters ≈ n_layer × 12 × d²**
(the embeddings and biases are rounding error once `d` is large). Scaling `d` or
`n_layer` is how you climb from 211K to billions. Run `python -c "..."` over the
checkpoint and you can reproduce every number above.

---

<a id="training"></a>
## 8. How training actually worked (`train.py`)

The full training loop is ~30 lines. Here is precisely what happens.

**Hyperparameters**
```
batch_size    = 32
block_size    = 128       # characters of context per example
max_iters     = 3000
learning_rate = 3e-3
optimizer     = AdamW
seed          = 1337
device        = CPU
```

**1. Encode the corpus** into one long tensor of character IDs:
```python
data = torch.tensor([stoi[c] for c in text])   # shape: (4096,)
```

**2. Sample mini-batches.** Each step we grab 32 random windows of 129 characters.
The first 128 are the input `x`; the same window shifted by one is the target `y`.
So for every input character, the label is simply *the next character*:
```python
ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
x  = torch.stack([data[i   : i+block_size]   for i in ix])
y  = torch.stack([data[i+1 : i+1+block_size] for i in ix])
```

**3. Forward + loss.** The model outputs logits of shape `(batch, 128, 54)`. We
compare them to the true next characters with **cross-entropy loss** — the standard
"how surprised was the model by the right answer?" measure.

**4. Backprop + update.** `loss.backward()` computes gradients; `AdamW` nudges every
parameter a little to reduce the loss. Repeat 3000 times.

**What the loss curve looked like (real run):**
```
iter     1 | loss 3.9995   <- random guessing (≈ ln(54) = 3.99 nats)
iter   500 | loss 1.1311
iter  1000 | loss 0.3639
iter  1500 | loss 0.2585
iter  2000 | loss 0.1897
iter  3000 | loss 0.1450   <- has essentially memorised the page
```

The starting loss of ~3.99 is not a coincidence: with 54 equally-likely characters,
a model that knows nothing has loss `ln(54) ≈ 3.99`. Watching it fall to ~0.15 is
watching the model memorise its 4 KB. **On a real LLM you would fight to *prevent*
this overfitting; here we welcome it**, because a model that has perfectly memorised
one page of satire is exactly the bit. Training takes a few seconds on a CPU.

The result is saved to `ckpt.pt` (~1.1 MB) — the model's entire brain, including the
character mappings, so inference needs nothing else.

---

<a id="inference"></a>
## 9. How it answers questions (inference, `app.py`)

The model only predicts next characters — it has no idea what a "question" is. We get
chat-like behaviour with a small prompt trick:

1. Wrap the user's text in the same shape it saw during training:
   `prompt = f"Q: {question}\nA:"`.
2. Feed it in and **autoregressively sample** one character at a time, appending each
   to the context, until the model emits a newline (end of the answer) or hits a
   length cap.
3. Two sampling knobs control the vibe:
   - **temperature** — divides the logits before softmax. Low (`0.3`) = safe and
     repetitive; high (`1.5`) = wilder and more typo-prone ("Step threee, raise").
   - **top-k** — only sample from the `k` most likely next characters, which keeps
     output from going totally random.

Because it has memorised the corpus, a known question returns its answer verbatim,
and a *novel* question gets snapped to the nearest memorised groove — confident,
on-brand nonsense. That is the entire personality of the model.

---

<a id="lessons"></a>
## 10. What this teaches you about real LLMs

Everything here scales up, conceptually unchanged:

- **Same objective.** GPT-4 and SLM 4096 are both just next-token predictors trained
  with cross-entropy. Scale and data are the difference, not the goal.
- **Parameters ≈ n_layer × 12 × d².** This is why "make it bigger" means widening `d`
  and stacking more layers.
- **Overfitting is the central tension.** Real models have *so much* data that they
  generalise instead of memorise. We removed the data to expose the mechanism.
- **What we left out is what makes ChatGPT useful:** BPE tokenization, massive
  diverse data, long context, and **post-training (SFT + RLHF)** that turns a raw
  next-token predictor into a helpful assistant. SLM 4096 is the raw predictor with
  none of that — which is why it's funny instead of helpful.

If you understand this repo, you understand the *skeleton* of every modern LLM.

---

<a id="run"></a>
## 11. Run / retrain it yourself

```bash
# 1. environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy fastapi "uvicorn[standard]"

# 2. (re)train — edit corpus.txt first to change its personality
.venv/bin/python train.py            # writes ckpt.pt, prints the loss curve

# 3. serve locally
.venv/bin/uvicorn app:app --port 7860
# open http://localhost:7860
```

**Things to try (great learning exercises):**
- Edit `corpus.txt` and retrain — watch the personality change.
- Change `n_embd` or `n_layer` in `train.py` and re-run the param count.
- Set `dropout = 0.0` and watch it memorise even harder.
- Lower `max_iters` and see a less-trained, more-garbled model.

---

<a id="files"></a>
## 12. File tour

| File | What it is |
|---|---|
| `corpus.txt` | The entire training set — exactly 4096 bytes of `Q:`/`A:` satire. |
| `model.py` | Self-contained minGPT char-level transformer (~150 lines). |
| `train.py` | The training loop; produces `ckpt.pt`. |
| `ckpt.pt` | Trained weights + character mappings (the whole brain, ~1.1 MB). |
| `app.py` | FastAPI server: serves the site and the `/generate` + `/meta` API. |
| `frontend/` | The editorial landing page, cursor-reactive particle field, chat UI. |
| `Dockerfile` | Builds the CPU image for Hugging Face Spaces. |

---

<a id="deploy"></a>
## 13. Deploy (Hugging Face Spaces, free CPU tier)

This repo is a ready-to-run **Docker** Space. In short:

1. Create a free account at huggingface.co and a **Write** access token.
2. Create a new **Docker** Space named `slm-4096` (CPU basic / free, public).
3. Push this code to the Space's git remote. It builds and serves itself at
   `https://<username>-slm-4096.hf.space`.

The model runs server-side in the container; visitors just need a browser. No GPU,
no API keys, no cost.

---

## Disclaimer

A parody and a teaching toy. Any resemblance to a useful product is coincidental.
Not affiliated with any AI lab. Smaller is the new bigger.
