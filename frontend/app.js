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

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.textContent;
      ask(chip.textContent);
    });
  });
})();
