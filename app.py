"""
SLM 4096 - FastAPI backend.

Serves the editorial landing page (frontend/) and a tiny inference API that runs
the 4 KB char-level model. Everything is local + free: no external model calls.

Run locally:  uvicorn app:app --reload --port 7860
Then open:    http://localhost:7860
"""

import os

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "ckpt.pt")
FRONTEND_DIR = os.path.join(HERE, "frontend")

device = "cpu"

# --- load the entire model brain (also tiny) ----------------------------------
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

stoi = ckpt["stoi"]
itos = {int(k): v for k, v in ckpt["itos"].items()}
CORPUS_BYTES = ckpt.get("corpus_bytes", 4096)
N_PARAMS = ckpt.get("n_params", model.num_params())
NEWLINE_ID = stoi.get("\n")
UNK_ID = stoi.get(" ", 0)  # fall back to space for unseen characters


def encode(s: str):
    return [stoi.get(ch, UNK_ID) for ch in s]


def generate_answer(question: str, temperature: float = 0.8, top_k: int = 20) -> str:
    question = (question or "").strip().replace("\n", " ")[:160]
    prompt = f"Q: {question}\nA:"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    # clamp so the prompt never exceeds the block size
    idx = idx[:, -cfg.block_size:]
    stop_ids = {NEWLINE_ID} if NEWLINE_ID is not None else None
    out = model.generate(
        idx,
        max_new_tokens=160,
        temperature=max(0.1, min(float(temperature), 2.0)),
        top_k=top_k,
        stop_ids=stop_ids,
    )
    full = "".join(itos[i] for i in out[0].tolist())
    # take everything after the final "A:" prompt, first line only
    answer = full.split("A:", 1)[-1].split("\n", 1)[0].strip()
    return answer or "..."


app = FastAPI(title="SLM 4096", docs_url=None, redoc_url=None)


class GenRequest(BaseModel):
    question: str = ""
    temperature: float = 0.8


@app.get("/meta")
def meta():
    return {
        "params": N_PARAMS,
        "corpus_bytes": CORPUS_BYTES,
        "vocab": cfg.vocab_size,
        "block_size": cfg.block_size,
        "layers": cfg.n_layer,
        "heads": cfg.n_head,
        "embd": cfg.n_embd,
    }


@app.post("/generate")
def generate(req: GenRequest):
    answer = generate_answer(req.question, req.temperature)
    return JSONResponse({"answer": answer})


# serve the static frontend at "/" (mounted last so /meta and /generate win)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
