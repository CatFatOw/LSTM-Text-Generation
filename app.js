const form = document.querySelector("#generator-form");
const promptInput = document.querySelector("#prompt");
const lengthInput = document.querySelector("#length");
const temperatureInput = document.querySelector("#temperature");
const output = document.querySelector("#output");
const copyButton = document.querySelector("#copy-button");
const generateButton = document.querySelector("#generate-button");
const modelStatus = document.querySelector("#model-status");
const weightsForm = document.querySelector("#weights-form");
const weightsInput = document.querySelector("#weights");
const weightsButton = document.querySelector("#weights-button");
const uploadMessage = document.querySelector("#upload-message");

let lastGeneratedText = "";

const demoWords = [
  "the", "lantern", "chapter", "house", "remembered", "softly", "garden",
  "window", "letter", "river", "evening", "voice", "secret", "paper",
  "almost", "again", "where", "Anna", "stood", "before", "door", "silver",
  "quiet", "promise", "inside", "morning", "road", "watching", "story",
  "name", "long", "little", "world", "turned", "away", "light", "found",
  "between", "until", "dream", "said", "knew", "old", "room", "held"
];

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[char]));
}

function formatGeneratedText(text) {
  let formatted = text.replace(/\s+([,.:;?!%)])/g, "$1");
  formatted = formatted.replace(/([(])\s+/g, "$1");
  formatted = formatted.replace(/\s+/g, " ").trim();

  const parts = formatted.split(/([.!?]\s+)/);
  let capitalized = "";
  let capitalizeNext = true;

  for (const part of parts) {
    let segment = part;
    if (capitalizeNext && segment.length > 0) {
      segment = segment.slice(0, 1).toUpperCase() + segment.slice(1);
      capitalizeNext = false;
    }
    capitalized += segment;
    if (/[.!?]\s+$/.test(segment)) {
      capitalizeNext = true;
    }
  }

  const sentences = capitalized.split(/(?<=[.!?])\s+/);
  const paragraphs = [];
  for (let i = 0; i < sentences.length; i += 4) {
    paragraphs.push(sentences.slice(i, i + 4).join(" "));
  }
  return paragraphs.join("\n\n");
}

function demoGenerate(prompt, length, temperature) {
  const seed = prompt.toLowerCase().split(/\s+/).filter(Boolean);
  const words = seed.length ? [...seed] : ["anna"];
  const punctuation = [".", ",", "", "", "", "?", "", ";"];
  const temperatureBias = Math.max(1, Math.round(5 / Number(temperature)));

  while (words.length < length) {
    const previous = words[words.length - 1] || "";
    const base = previous.charCodeAt(0) || words.length;
    const index = Math.abs(base + words.length * 17 + prompt.length * 3) % demoWords.length;
    const drift = Math.floor(Math.random() * Math.max(2, temperatureBias + 3));
    let next = demoWords[(index + drift) % demoWords.length];
    if (words.length % 19 === 0) next += punctuation[Math.floor(Math.random() * punctuation.length)];
    if (words.length % 47 === 0) next += ".";
    words.push(next);
  }

  return formatGeneratedText(words.join(" "));
}

function renderText(text) {
  lastGeneratedText = text;
  output.innerHTML = text
    .split(/\n{2,}/)
    .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
    .join("");
  copyButton.disabled = !text;
}

async function updateStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Status unavailable");
    const status = await response.json();
    modelStatus.dataset.mode = status.ready ? "ready" : "demo";
    modelStatus.querySelector("span:last-child").textContent = status.ready ? "LSTM ready" : "Demo mode";
    if (status.checkpoint) {
      uploadMessage.textContent = `Active checkpoint: ${status.checkpoint}`;
    }
  } catch {
    modelStatus.dataset.mode = "demo";
    modelStatus.querySelector("span:last-child").textContent = "Static demo";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  const length = Number(lengthInput.value) || 200;
  const temperature = Number(temperatureInput.value) || 1;
  if (!prompt) return;

  generateButton.disabled = true;
  generateButton.textContent = "Generating";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, length, temperature })
    });
    if (!response.ok) throw new Error("Backend unavailable");
    const data = await response.json();
    renderText(data.text);
    modelStatus.dataset.mode = data.mode === "model" ? "ready" : "demo";
    modelStatus.querySelector("span:last-child").textContent = data.mode === "model" ? "LSTM ready" : "Demo mode";
  } catch {
    renderText(demoGenerate(prompt, length, temperature));
    modelStatus.dataset.mode = "demo";
    modelStatus.querySelector("span:last-child").textContent = "Static demo";
  } finally {
    generateButton.disabled = false;
    generateButton.innerHTML = '<span class="button-icon" aria-hidden="true">+</span> Generate';
  }
});

weightsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = weightsInput.files && weightsInput.files[0];
  if (!file) {
    uploadMessage.textContent = "Choose a weights file first.";
    return;
  }

  weightsButton.disabled = true;
  weightsButton.textContent = "Uploading";
  uploadMessage.textContent = "Uploading checkpoint...";

  const body = new FormData();
  body.append("weights", file);

  try {
    const response = await fetch("/api/upload-weights", {
      method: "POST",
      body
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Upload failed");

    modelStatus.dataset.mode = data.ready ? "ready" : "demo";
    modelStatus.querySelector("span:last-child").textContent = data.ready ? "LSTM ready" : "Demo mode";
    uploadMessage.textContent = data.message;
  } catch (error) {
    uploadMessage.textContent = error.message || "Upload failed.";
  } finally {
    weightsButton.disabled = false;
    weightsButton.textContent = "Upload";
  }
});

copyButton.addEventListener("click", async () => {
  if (!lastGeneratedText) return;
  await navigator.clipboard.writeText(lastGeneratedText);
  copyButton.textContent = "Copied";
  setTimeout(() => {
    copyButton.textContent = "Copy";
  }, 1200);
});

updateStatus();
