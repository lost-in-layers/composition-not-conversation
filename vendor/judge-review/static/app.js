"use strict";

const $ = (id) => document.getElementById(id);
const state = { reviewer: "", position: 0, total: 0, done: 0, item: null };

function say(message, isError) {
  const box = $("message");
  box.textContent = message || "";
  box.style.color = isError ? "var(--red)" : "var(--muted)";
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (error) {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return response.json();
}

// ---------------------------------------------------------------- transcript

function lightbox(source) {
  const overlay = document.createElement("div");
  overlay.className = "lightbox";
  const image = document.createElement("img");
  image.src = source;
  overlay.appendChild(image);
  overlay.addEventListener("click", () => overlay.remove());
  document.body.appendChild(overlay);
}

function renderTurn(turn) {
  const row = document.createElement("div");
  row.className = `turn ${turn.role}`;

  const role = document.createElement("div");
  role.className = "role";
  role.textContent = turn.role;
  row.appendChild(role);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const text = turn.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n\n")
    .trim();
  bubble.textContent = text || "(no text)";

  const images = turn.parts.filter((part) => part.type === "image" && part.path);
  if (images.length) {
    const strip = document.createElement("div");
    strip.className = "shots";
    images.forEach((part) => {
      const source = `/api/image/${part.path}`;
      const cell = document.createElement("div");
      cell.className = "shot";
      const image = document.createElement("img");
      image.src = source;
      image.loading = "lazy";
      image.alt = part.label || "image";
      image.addEventListener("click", () => lightbox(source));
      const caption = document.createElement("div");
      caption.className = "cap";
      caption.textContent = part.label || "";
      cell.append(image, caption);
      strip.appendChild(cell);
    });
    bubble.appendChild(strip);
  }

  row.appendChild(bubble);
  return row;
}

function render(payload) {
  const item = payload.item;
  state.item = item;
  state.position = payload.position;
  state.total = payload.total;

  $("position").textContent = `${payload.position + 1} / ${payload.total}`;
  $("meta").textContent = `${item.condition} · ${item.model} · ${item.category}`;
  $("question").textContent = item.question || "(question unavailable)";
  $("reference").textContent = item.reference_answer || "(answer unavailable)";

  const transcript = $("transcript");
  transcript.replaceChildren();
  if (!item.trace || !item.trace.length) {
    const empty = document.createElement("div");
    empty.className = "bubble";
    empty.textContent = "No transcript was found. Mark this Unclear.";
    transcript.appendChild(empty);
  } else {
    item.trace.forEach((turn) => transcript.appendChild(renderTurn(turn)));
  }

  const existing = payload.existing;
  $("existing").textContent = existing
    ? `You marked this ${existing.verdict}. Deciding again replaces it.`
    : "";
  $("note").value = existing ? existing.note || "" : "";
  $("previousButton").disabled = payload.position === 0;
  say("");
  window.scrollTo({ top: 0, behavior: "instant" });
}

function progress() {
  $("doneCount").textContent = state.done;
  $("totalCount").textContent = state.total;
  $("goalFill").style.width = state.total
    ? `${(state.done / state.total) * 100}%`
    : "0%";
}

// -------------------------------------------------------------------- flow

async function load(position) {
  if (position < 0 || position >= state.total) return;
  const payload = await api(
    `/api/item/${position}?reviewer=${encodeURIComponent(state.reviewer)}`,
  );
  render(payload);
}

async function decide(verdict) {
  if (!state.item) return;
  $("saveState").textContent = "saving...";
  try {
    const result = await api("/api/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviewer: state.reviewer,
        index: state.item.index,
        verdict,
        note: $("note").value,
      }),
    });
    state.done = result.done;
    progress();
    $("saveState").textContent = `saved · ${result.export}`;
    if (state.position + 1 < state.total) {
      await load(state.position + 1);
    } else {
      say("Every conversation has a verdict. Thank you.");
    }
  } catch (error) {
    $("saveState").textContent = "not saved";
    say(error.message, true);
  }
}

async function start(name) {
  const session = await api(`/api/session?reviewer=${encodeURIComponent(name)}`);
  state.reviewer = session.reviewer;
  state.total = session.total;
  state.done = session.done;
  localStorage.setItem("judgeReviewName", state.reviewer);

  $("gate").hidden = true;
  $("bar").hidden = false;
  $("main").hidden = false;
  $("who").textContent = state.reviewer;
  $("help").open = session.done === 0;
  $("jumpInput").max = session.total;
  progress();
  await load(session.cursor);
}

// ------------------------------------------------------------------- wiring

$("gateForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = $("gateName").value.trim();
  if (!name) return;
  try {
    await start(name);
  } catch (error) {
    $("gateMessage").textContent = error.message;
  }
});

$("correctButton").addEventListener("click", () => decide("correct"));
$("wrongButton").addEventListener("click", () => decide("wrong"));
$("unclearButton").addEventListener("click", () => decide("unclear"));
$("previousButton").addEventListener("click", () => load(state.position - 1));
$("helpButton").addEventListener("click", () => {
  $("help").open = !$("help").open;
  $("help").scrollIntoView({ behavior: "smooth", block: "start" });
});
$("exportButton").addEventListener("click", () => {
  window.location.href = "/api/export/download";
});
$("jumpForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const wanted = Number($("jumpInput").value);
  if (wanted >= 1 && wanted <= state.total) load(wanted - 1);
});

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT" || $("gate").hidden === false) return;
  if (event.key === "1") decide("correct");
  if (event.key === "2") decide("wrong");
  if (event.key === "3") decide("unclear");
  if (event.key === "ArrowLeft") load(state.position - 1);
});

(async function boot() {
  try {
    const session = await api("/api/session?reviewer=__probe__");
    if (session.reviewers.length) {
      $("gateReviewers").innerHTML =
        "So far: " +
        session.reviewers
          .filter((entry) => entry.reviewer !== "__probe__")
          .map((entry) => `<b>${entry.reviewer}</b> (${entry.n_done})`)
          .join(", ");
    }
  } catch (error) {
    $("gateMessage").textContent = error.message;
  }
  const remembered = localStorage.getItem("judgeReviewName");
  if (remembered) $("gateName").value = remembered;
})();
