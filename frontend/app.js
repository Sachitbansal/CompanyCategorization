const STEP_LABELS = {
  cache_check: "Checking for an existing record",
  scrape: "Scraping website",
  summarize: "Generating company summary",
  retrieve_similar: "Retrieving similar companies",
  assign_tags: "Assigning tags",
  dedup_tags: "Running dedup guardrail",
  save: "Saving to database",
};

const input = document.getElementById("input");
const submitBtn = document.getElementById("submit-btn");
const logPanel = document.getElementById("log-panel");
const contextBox = document.getElementById("context-box");
const contextBrand = document.getElementById("context-brand");
const contextDraft = document.getElementById("context-draft");
const contextInput = document.getElementById("context-input");
const contextSubmit = document.getElementById("context-submit");
const contextSkip = document.getElementById("context-skip");
const resultCard = document.getElementById("result-card");

let socket = null;
let pendingBrand = "";

function logLine(text, kind) {
  const line = document.createElement("div");
  line.className = `log-line log-${kind}`;
  line.innerHTML = text;
  logPanel.appendChild(line);
  logPanel.scrollTop = logPanel.scrollHeight;
}

function resetRun() {
  logPanel.innerHTML = "";
  logPanel.classList.add("visible");
  contextBox.classList.remove("visible");
  resultCard.classList.remove("visible");
}

function setBusy(busy) {
  submitBtn.disabled = busy;
  submitBtn.innerHTML = busy ? '<span class="spinner"></span>Working...' : "Categorize";
}

function badge(text, extraClass) {
  const span = document.createElement("span");
  span.className = `badge ${extraClass || ""}`.trim();
  span.textContent = text;
  return span;
}

function renderResult(result) {
  resultCard.classList.add("visible");

  const cacheWrap = document.getElementById("cache-note-wrap");
  cacheWrap.innerHTML = "";
  if (result.cache_hit) {
    const note = document.createElement("div");
    note.className = "cache-note";
    note.textContent = "● Found an existing record — no LLM calls were made";
    cacheWrap.appendChild(note);
  }

  document.getElementById("result-name").textContent = result.name;
  document.getElementById("result-website").textContent = result.website || "";

  const kwLine = document.getElementById("result-keywords");
  kwLine.textContent = result.keywords && result.keywords.length ? result.keywords.join(", ") : "none";

  const primRow = document.getElementById("result-primary");
  primRow.innerHTML = "";
  primRow.appendChild(badge(result.primary_tag || "none", "primary"));

  const secRow = document.getElementById("result-secondary");
  secRow.innerHTML = "";
  if (result.secondary_tags && result.secondary_tags.length) {
    result.secondary_tags.forEach((t) => secRow.appendChild(badge(t, "secondary")));
  } else {
    secRow.appendChild(badge("none"));
  }
}

function handleEvent(event) {
  const label = STEP_LABELS[event.step] || event.step;

  if (event.step === "error") {
    logLine(`✕ ${event.detail || "something went wrong"}`, "error");
    setBusy(false);
    return;
  }

  if (event.step === "awaiting_context") {
    pendingBrand = event.brand_name;
    contextBrand.textContent = event.brand_name;
    contextDraft.textContent = `"${event.draft_summary}"`;
    contextBox.classList.add("visible");
    contextInput.value = "";
    contextInput.focus();
    setBusy(false);
    return;
  }

  if (event.step === "complete") {
    renderResult(event.result);
    setBusy(false);
    return;
  }

  if (event.status === "running") {
    logLine(`<span class="dot">○</span> <span class="step-name">${label}</span>...`, "running");
  } else if (event.status === "done") {
    const detail = event.detail ? ` — ${event.detail}` : "";
    logLine(`<span class="dot">●</span> <span class="step-name">${label}</span>${detail}`, "done");

    // The retrieval step carries the actual matched companies, not just a
    // count — list them with their similarity scores so it's visible WHICH
    // companies became few-shot examples for the tagging call.
    if (event.step === "retrieve_similar") {
      const matches = event.matches || [];
      if (matches.length) {
        matches.forEach((m) => {
          const score = typeof m.similarity === "number" ? m.similarity.toFixed(3) : "?";
          logLine(`<span class="sub">└ ${escapeHtml(m.name)} · ${score}</span>`, "sub");
        });
      } else {
        logLine(`<span class="sub">└ nothing cleared the threshold — no few-shot examples sent</span>`, "sub");
      }
    }
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function ensureSocket() {
  if (socket && socket.readyState === WebSocket.OPEN) return Promise.resolve(socket);

  return new Promise((resolve, reject) => {
    socket = new WebSocket(`${WS_BASE}/ws/categorize`);
    socket.onopen = () => resolve(socket);
    socket.onerror = (err) => reject(err);
    socket.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
    socket.onclose = () => {
      socket = null;
    };
  });
}

async function startCategorize() {
  const value = input.value.trim();
  if (!value) return;

  resetRun();
  setBusy(true);

  try {
    const ws = await ensureSocket();
    ws.send(JSON.stringify({ name_or_url: value }));
  } catch (err) {
    logLine("✕ Could not reach the backend — is it running?", "error");
    setBusy(false);
  }
}

async function resumeWithContext(extraContext) {
  contextBox.classList.remove("visible");
  setBusy(true);
  logLine(`<span class="dot">○</span> Resuming with your description...`, "running");
  try {
    const ws = await ensureSocket();
    ws.send(JSON.stringify({ extra_context: extraContext }));
  } catch (err) {
    logLine("✕ Could not reach the backend — is it running?", "error");
    setBusy(false);
  }
}

submitBtn.addEventListener("click", startCategorize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") startCategorize();
});

contextSubmit.addEventListener("click", () => resumeWithContext(contextInput.value.trim()));
contextSkip.addEventListener("click", () => resumeWithContext(""));
