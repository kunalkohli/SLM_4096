"""
Train SLM 4096 on exactly one page of RAM (4096 bytes of corpus.txt).

Run locally:  python train.py
Produces:     ckpt.pt  (the entire model brain, also tiny)

With only 4 KB of data the model overfits/memorises by design - that is the joke.
It learns the Q:/A: pattern and remixes the satire into confident nonsense.
"""

import json
import torch

from model import GPT, GPTConfig

# --- hyperparameters (small on purpose) ---------------------------------------
BLOCK_SIZE = 128
N_LAYER = 4
N_HEAD = 4
N_EMBD = 64
DROPOUT = 0.1
BATCH_SIZE = 32
MAX_ITERS = 3000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-3
SEED = 1337
CORPUS_PATH = "corpus.txt"
CKPT_PATH = "ckpt.pt"


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = open(CORPUS_PATH, "r", encoding="utf-8").read()
    raw_bytes = len(text.encode("utf-8"))
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"corpus: {raw_bytes} bytes | vocab: {vocab_size} chars")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    def get_batch():
        ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
        return x.to(device), y.to(device)

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for it in range(1, MAX_ITERS + 1):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if it % EVAL_INTERVAL == 0 or it == 1:
            print(f"iter {it:5d} | loss {loss.item():.4f}")

    ckpt = {
        "model_state": model.state_dict(),
        "config": vars(cfg),
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "corpus_bytes": raw_bytes,
        "n_params": n_params,
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"saved {CKPT_PATH}")

    # quick sanity sample
    model.eval()
    prompt = "Q: What is SLM?\nA:"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=120, temperature=0.8, top_k=20)
    text_out = "".join(itos[i] for i in out[0].tolist())
    print("\n--- sample ---")
    print(text_out)


if __name__ == "__main__":
    main()
