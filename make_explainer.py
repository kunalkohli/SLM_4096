"""
Generates the "neural network vs transformer" diagram (inline SVG, brand-matched).

Outputs:
  * frontend/explainer.html         - standalone page
  * injects the SVG + key takeaways into frontend/index.html between the
    <!-- DIAGRAM:START --> / <!-- DIAGRAM:END --> markers (single source of truth)
"""

CREAM = "#f0eee6"; CREAM2 = "#e9e6db"; INK = "#1a1a1a"; INKS = "#4a463d"
ACCENT = "#c15f3c"; ACCENT_T = "#f3e0d7"; RULE = "#cfc9ba"

S = []
def add(x): S.append(x)

def text(x, y, s, size=14, fill=INK, family="mono", anchor="middle", weight="400", style="normal", ls="0"):
    fam = "'Space Mono', monospace" if family == "mono" else "'Fraunces', Georgia, serif"
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" '
            f'letter-spacing="{ls}">{s}</text>')

def box(x, y, w, h, fill=CREAM2, stroke=INK, sw=1.4, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def arrow(x1, y1, x2, y2, color=INK, w=1.8):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" marker-end="url(#ah)"/>'

def circle(cx, cy, r, fill=CREAM2, stroke=INK, sw=1.6):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

# ---------- defs ----------
add('<defs>'
    f'<marker id="ah" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{INK}"/></marker>'
    f'<marker id="aha" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L6,3 L0,6 Z" fill="{ACCENT}"/></marker>'
    '</defs>')

# ---------- headers ----------
add(text(330, 70, "A PLAIN NEURAL NETWORK", size=15, fill=INK, ls="2"))
add(text(330, 94, "(multi-layer perceptron)", size=12, fill=INKS, family="serif", style="italic"))
add(text(945, 70, "A TRANSFORMER  \u00b7  OUR SLM", size=15, fill=ACCENT, ls="2"))
add(text(945, 94, "(a neural network + self-attention)", size=12, fill=INKS, family="serif", style="italic"))
add(f'<line x1="650" y1="58" x2="650" y2="604" stroke="{RULE}" stroke-width="1.5"/>')

# ================= LEFT: MLP =================
def column(cx, n, cy=340, gap=58):
    return [(cx, cy + (i - (n - 1) / 2) * gap) for i in range(n)]

L0 = column(150, 4); L1 = column(300, 6); L2 = column(440, 6); L3 = column(560, 2)
for a, nxt in ((L0, L1), (L1, L2), (L2, L3)):
    for (x1, y1) in a:
        for (x2, y2) in nxt:
            add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="0.8" stroke-opacity="0.16"/>')
for (x, y) in L0: add(circle(x, y, 14, fill=CREAM, stroke=ACCENT, sw=2))
for (x, y) in L1: add(circle(x, y, 14))
for (x, y) in L2: add(circle(x, y, 14))
for (x, y) in L3: add(circle(x, y, 14, fill=ACCENT_T, stroke=ACCENT, sw=2))
add(text(150, 600, "input", size=12, fill=INKS))
add(text(370, 600, "hidden layers", size=12, fill=INKS))
add(text(560, 600, "output", size=12, fill=INKS))
add(text(330, 642, "Fixed-size input \u2192 multiply by learned weights \u2192", size=13, fill=INKS, family="serif"))
add(text(330, 664, "nonlinearity \u2192 repeat. Every input is processed on its", size=13, fill=INKS, family="serif"))
add(text(330, 686, "own. No notion of order, and inputs never interact.", size=13, fill=INKS, family="serif"))

# ================= RIGHT: TRANSFORMER =================
cx = 880; bw = 300; x0 = cx - bw / 2

def vbox(y, h, title, subtitle=None, kind="plain"):
    if kind == "attn":
        add(box(x0, y, bw, h, fill=ACCENT_T, stroke=ACCENT, sw=2.2))
    elif kind == "out":
        add(box(x0, y, bw, h, fill=CREAM, stroke=INK))
    else:
        add(box(x0, y, bw, h))
    tcol = ACCENT if kind == "attn" else INK
    add(text(cx, y + (24 if subtitle else h / 2 + 5), title, size=15, fill=tcol, family="serif", weight="600"))
    if subtitle:
        add(text(cx, y + 44, subtitle, size=11.5, fill=INKS, family="mono"))

vbox(120, 46, "Next-character probabilities", kind="out")
vbox(192, 46, "Linear head  \u2192  Softmax")
# block container + tag (tag sized to fully contain the text)
add(box(x0 - 16, 262, bw + 32, 236, fill="none", stroke=ACCENT, sw=1.4, rx=12))
tag_w = 204
add(f'<rect x="{x0-16}" y="262" width="{tag_w}" height="26" rx="8" fill="{ACCENT}"/>')
add(text(x0 - 16 + 14, 280, "TRANSFORMER BLOCK \u00d74", size=11, fill=CREAM, anchor="start", ls="1.5"))
vbox(300, 66, "Feed-Forward Network", "a plain neural net (MLP)")
add(text(cx, 384, "+ Add &amp; LayerNorm", size=10.5, fill=INKS))
vbox(398, 84, "Multi-Head Self-Attention", "causal: each token looks back", kind="attn")
vbox(524, 50, "Token + Positional Embeddings")

toks = list("SLM4096")
tx0 = cx - (len(toks) * 38) / 2 + 19
for i, ch in enumerate(toks):
    bx = tx0 + i * 38 - 15
    add(box(bx, 588, 30, 30, fill=CREAM, stroke=INK, rx=5))
    add(text(bx + 15, 609, ch, size=14, fill=INK))
add(text(cx, 640, "input characters \u2192 numbers (one per character)", size=12, fill=INKS))

for (ya, yb) in [(588, 576), (524, 500), (262, 240), (192, 170)]:
    add(arrow(cx, ya, cx, yb))
add(arrow(cx, 398, cx, 368, color=ACCENT, w=1.8))

# attention inset
ix = [1058, 1108, 1158, 1208]; iy = 440
add(text(1133, 392, "every token attends", size=11, fill=ACCENT))
add(text(1133, 408, "to earlier tokens", size=11, fill=ACCENT))
for x in ix:
    add(circle(x, iy, 9, fill=CREAM, stroke=ACCENT, sw=1.8))
for j in range(len(ix)):
    for k in range(j):
        x1, x2 = ix[k], ix[j]
        midx = (x1 + x2) / 2; lift = 26 + (j - k) * 8
        add(f'<path d="M{x1},{iy-9} Q{midx},{iy-lift} {x2},{iy-9}" fill="none" stroke="{ACCENT}" stroke-width="1.2" stroke-opacity="0.75" marker-end="url(#aha)"/>')
add(f'<line x1="1013" y1="440" x2="1040" y2="440" stroke="{ACCENT}" stroke-width="1.2" stroke-dasharray="3,3"/>')

add(text(880, 680, "Same neural-net parts (linear layers, nonlinearities) PLUS self-", size=13, fill=INKS, family="serif"))
add(text(880, 702, "attention, so every token mixes in the context of the others.", size=13, fill=INKS, family="serif"))

svg = ('<svg viewBox="0 0 1280 740" xmlns="http://www.w3.org/2000/svg" role="img" '
       'aria-label="Diagram comparing a plain neural network with a transformer" '
       'style="width:100%;height:auto;display:block">' + "".join(S) + '</svg>')

# ---------- key takeaways (shared) ----------
KEYS = [
    ("Plain neural net",
     "Layers of neurons, fully connected by <b>learned weights</b>. Great for a fixed input "
     "(an image, a row of numbers). It has <b>no idea about order</b> and the inputs never look at each other."),
    ("Self-attention",
     "The new ingredient. Every token <b>looks at the other tokens</b> and pulls in what is relevant, "
     "so meaning depends on <b>context</b>. Positional embeddings add the sense of order."),
    ("So what is a transformer?",
     "A neural network <b>plus self-attention</b>. The feed-forward block inside each layer <b>is</b> "
     "an ordinary neural net \u2014 attention is simply wrapped around it, repeated a few times."),
]

def keys_html(htag, kcls):
    out = [f'<div class="{kcls}">']
    for title, body in KEYS:
        out.append(f'<div class="{kcls[:-1] if kcls.endswith("s") else kcls}-item"><{htag}>{title}</{htag}><p>{body}</p></div>')
    out.append('</div>')
    return "".join(out)

# ---------- standalone page ----------
standalone_keys = "".join(
    f'<div class="key"><h3>{t}</h3><p>{b}</p></div>' for t, b in KEYS)
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Neural Network vs Transformer — SLM 4096</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,400&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet" />
<style>
  body {{ margin:0; background:{CREAM}; color:{INK}; font-family:'Fraunces',Georgia,serif; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1240px; margin:0 auto; padding:48px 28px 80px; }}
  .eyebrow {{ font-family:'Space Mono',monospace; font-size:12px; letter-spacing:2.5px; color:{INKS}; display:flex; align-items:center; gap:14px; }}
  .eyebrow .rule {{ width:46px; height:1px; background:{ACCENT}; display:inline-block; }}
  h1 {{ font-size:clamp(34px,5vw,60px); font-weight:600; letter-spacing:-1px; margin:18px 0 14px; }}
  h1 em {{ font-style:italic; color:{ACCENT}; font-weight:400; }}
  .lede {{ font-size:19px; line-height:1.55; color:{INKS}; max-width:760px; margin:0 0 36px; }}
  .card {{ background:{CREAM2}; border:1px solid {RULE}; border-radius:10px; padding:34px; overflow-x:auto; }}
  .keys {{ display:grid; grid-template-columns:repeat(3,1fr); gap:24px; margin-top:40px; }}
  .key h3 {{ font-family:'Space Mono',monospace; font-size:12px; letter-spacing:1.5px; text-transform:uppercase; color:{ACCENT}; border-bottom:1px solid {RULE}; padding-bottom:10px; margin:0 0 12px; }}
  .key p {{ font-size:15px; line-height:1.55; color:{INKS}; margin:0; }}
  .key b {{ color:{INK}; font-weight:600; }}
  @media (max-width:760px){{ .keys{{grid-template-columns:1fr;}} }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="eyebrow"><span class="rule"></span> SLM 4096 &middot; UNDER THE HOOD</div>
    <h1>Neural network <em>vs</em> transformer</h1>
    <p class="lede">A transformer <em>is</em> a neural network — but a special kind. Here is the
      plain version next to the transformer our SLM uses, with the one part that makes the
      difference highlighted.</p>
    <div class="card">{svg}</div>
    <div class="keys">{standalone_keys}</div>
  </div>
</body>
</html>
"""
with open("frontend/explainer.html", "w", encoding="utf-8") as f:
    f.write(html)

# ---------- inject into index.html ----------
inline_keys = "".join(
    f'<div class="nn-key"><h4>{t}</h4><p>{b}</p></div>' for t, b in KEYS)
fragment = (f'      <div class="diagram-card">{svg}</div>\n'
            f'      <div class="nn-keys">{inline_keys}</div>')

idx_path = "frontend/index.html"
doc = open(idx_path, encoding="utf-8").read()
START = "<!-- DIAGRAM:START -->"; END = "<!-- DIAGRAM:END -->"
pre, rest = doc.split(START, 1)
_, post = rest.split(END, 1)
doc = pre + START + "\n" + fragment + "\n      " + END + post
open(idx_path, "w", encoding="utf-8").write(doc)

print("wrote frontend/explainer.html and injected diagram into frontend/index.html")
