/* SLM 4096 - frontend chat wiring + live stats from the backend. */
(function () {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("question");
  const askBtn = document.getElementById("ask-btn");
  const answerText = document.getElementById("answer-text");
  const temp = document.getElementById("temp");
  const tempVal = document.getElementById("temp-val");

  // live temperature label
  temp.addEventListener("input", () => { tempVal.textContent = temp.value; });

  // pull real model stats from /meta
  fetch("/meta")
    .then((r) => r.json())
    .then((m) => {
      const p = document.getElementById("stat-params");
      const b = document.getElementById("stat-bytes");
      if (p && m.params) p.textContent = m.params.toLocaleString();
      if (b && m.corpus_bytes) b.textContent = m.corpus_bytes + " bytes";
    })
    .catch(() => {});

  function typeOut(text) {
    answerText.classList.remove("muted");
    answerText.classList.add("cursor-blink");
    answerText.textContent = "";
    let i = 0;
    const timer = setInterval(() => {
      answerText.textContent = text.slice(0, i);
      i++;
      if (i > text.length) {
        clearInterval(timer);
        answerText.classList.remove("cursor-blink");
      }
    }, 16);
  }

  async function ask(question) {
    if (!question.trim()) return;
    askBtn.disabled = true;
    answerText.classList.remove("muted");
    answerText.classList.add("cursor-blink");
    answerText.textContent = "thinking (briefly)…";
    try {
      const res = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          temperature: parseFloat(temp.value),
        }),
      });
      const data = await res.json();
      typeOut(data.answer || "…");
    } catch (e) {
      answerText.classList.remove("cursor-blink");
      answerText.textContent = "The model is asleep (free tier). Try again in a moment.";
    } finally {
      askBtn.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    ask(input.value);
  });

  // pool of example questions (drawn from the 4 KB the model actually knows)
  const QUESTION_POOL = [
    "Are you AGI?",
    "Can you replace my team?",
    "What is your moat?",
    "Is it secure?",
    "Do you hallucinate?",
    "How do you scale?",
    "Can you write code?",
    "What is your pricing?",
    "What about alignment?",
    "Do you have memory?",
    "Is it open source?",
    "What is your roadmap?",
    "Can you reason?",
    "What data were you trained on?",
    "How smart is it really?",
    "What is a transformer?",
    "Can you do math?",
    "Do you understand language?",
    "What is your accuracy?",
    "Is it multimodal?",
    "How do you handle bias?",
    "Do you use RAG?",
    "Can you be my agent?",
    "What is fine tuning?",
    "Why only four kilobytes?",
    "Is the future of AI small?",
    "Will you take my job?",
    "How is this disruptive?",
  ];

  const suggestions = document.getElementById("suggestions");
  const shuffleBtn = document.getElementById("shuffle-btn");

  function pickFour() {
    const pool = QUESTION_POOL.slice();
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, 4);
  }

  function renderChips() {
    suggestions.innerHTML = "";
    pickFour().forEach((q) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = q;
      chip.addEventListener("click", () => {
        input.value = q;
        ask(q);
      });
      suggestions.appendChild(chip);
    });
  }

  if (shuffleBtn) {
    shuffleBtn.addEventListener("click", () => {
      shuffleBtn.classList.add("spin");
      setTimeout(() => shuffleBtn.classList.remove("spin"), 400);
      renderChips();
    });
  }

  renderChips();
})();
