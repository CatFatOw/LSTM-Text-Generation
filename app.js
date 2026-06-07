const form = document.querySelector("#generator-form");
const promptInput = document.querySelector("#prompt");
const lengthInput = document.querySelector("#length");
const temperatureInput = document.querySelector("#temperature");
const topKInput = document.querySelector("#top-k");
const output = document.querySelector("#output");
const copyButton = document.querySelector("#copy-button");
const generateButton = document.querySelector("#generate-button");
const modelStatus = document.querySelector("#model-status");
const weightsForm = document.querySelector("#weights-form");
const weightsInput = document.querySelector("#weights");
const weightsButton = document.querySelector("#weights-button");
const uploadMessage = document.querySelector("#upload-message");
const examplePrompt = document.querySelector("#example-prompt");
const pageTitle = document.querySelector("#page-title");
const modelTitle = document.querySelector("#model-title");
const modelDescription = document.querySelector("#model-description");
const checkpointNote = document.querySelector("#checkpoint-note");
const bookTabs = Array.from(document.querySelectorAll(".book-tab"));

let lastGeneratedText = "";
let currentBook = "anna";
let statusByBook = {};

const books = {
  anna: {
    title: "Anna Karenina",
    bodyClass: "theme-anna",
    modelTitle: "Anna Karenina LSTM",
    checkpoint: "LSTM_Annie.pth",
    example: "Anna and the prince",
    description: "A society novel model tuned toward drawing rooms, glances, family tension, and Tolstoy's restless interior cadence.",
    demoWords: ["princess", "vronsky", "levin", "drawing", "room", "smile", "carriage", "letter", "heart", "dolly", "anna", "prince", "silence", "love", "moscow"]
  },
  war: {
    title: "The War of the Worlds",
    bodyClass: "theme-war",
    modelTitle: "War of the Worlds LSTM",
    checkpoint: "war_lstm.pth",
    example: "the martians attacked, bringing desctruction and death along its wake.",
    description: "A science-fiction invasion model built around Martians, cylinders, smoke, heat-rays, London roads, and planetary dread.",
    demoWords: ["martians", "cylinder", "woking", "heat", "ray", "smoke", "tripod", "london", "earth", "pit", "red", "weed", "destruction", "death", "darkness"]
  }
};

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
  const demoWords = books[currentBook].demoWords;
  const seed = prompt.toLowerCase().split(/\s+/).filter(Boolean);
  const words = seed.length ? [...seed] : [currentBook === "war" ? "martians" : "anna"];
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
    statusByBook = status.books || {};
    renderBookState();
  } catch {
    modelStatus.dataset.mode = "demo";
    modelStatus.querySelector("span:last-child").textContent = "Static demo";
  }
}

function renderBookState() {
  const book = books[currentBook];
  const status = statusByBook[currentBook] || {};
  document.body.classList.remove("theme-anna", "theme-war");
  document.body.classList.add(book.bodyClass);
  pageTitle.textContent = book.title;
  modelTitle.textContent = book.modelTitle;
  modelDescription.textContent = book.description;
  examplePrompt.textContent = `Example: ${book.example}`;
  checkpointNote.innerHTML = `Default checkpoint: <strong>${book.checkpoint}</strong>`;
  modelStatus.dataset.mode = status.ready ? "ready" : "demo";
  modelStatus.querySelector("span:last-child").textContent = status.ready ? "LSTM ready" : "Demo mode";
  uploadMessage.textContent = status.checkpoint ? `Active checkpoint: ${status.checkpoint}` : "";
  bookTabs.forEach((tab) => {
    const active = tab.dataset.book === currentBook;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  const length = Number(lengthInput.value) || 300;
  const temperature = Number(temperatureInput.value) || 0.3;
  const topK = Number(topKInput.value) || 3;
  if (!prompt) return;

  generateButton.disabled = true;
  generateButton.textContent = "Generating";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ book: currentBook, prompt, length, temperature, top_k: topK })
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
  body.append("book", currentBook);

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

bookTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    currentBook = tab.dataset.book;
    promptInput.value = "";
    renderText("");
    output.innerHTML = '<p class="placeholder">Your generated passage will appear here after you enter a prompt.</p>';
    renderBookState();
  });
});

examplePrompt.addEventListener("click", () => {
  promptInput.value = books[currentBook].example;
  promptInput.focus();
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
