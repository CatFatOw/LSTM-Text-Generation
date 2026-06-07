from __future__ import annotations

import json
import os
import re
import shutil
import ast
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from random import choice, random
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
BUNDLED_MODEL_PATH = ROOT / "models" / "LSTM_Annie.pth"
DOWNLOAD_MODEL_PATH = Path("/Users/michaelwu/Downloads/LSTM_Annie.pth")
HF_MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "CatFatOw123/Anna_Karenina_Model")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "LSTM_Annie.pth")
BUNDLED_OUTPUT_PATH = ROOT / "output.txt"
DOWNLOAD_OUTPUT_PATH = Path("/Users/michaelwu/Downloads/output.txt")
UPLOAD_DIR = ROOT / "uploaded_weights"
ACTIVE_MODEL_POINTER = UPLOAD_DIR / "active_checkpoint.txt"
SEQ_LEN = 100

MODEL: Any = None
TORCH: Any = None
DEVICE = "cpu"
WORD_TO_INT: dict[str, int] = {}
INT_TO_WORD: dict[int, str] = {}
MODEL_ERROR = ""
ACTIVE_MODEL_PATH = BUNDLED_MODEL_PATH


def default_model_path() -> Path:
    for path in (BUNDLED_MODEL_PATH, DOWNLOAD_MODEL_PATH):
        if path.exists():
            return path

    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=HF_MODEL_REPO_ID,
                filename=HF_MODEL_FILENAME,
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not find a local model file or download {HF_MODEL_REPO_ID}/{HF_MODEL_FILENAME}: {exc}"
        ) from exc


def active_model_path() -> Path:
    if ACTIVE_MODEL_POINTER.exists():
        saved = ACTIVE_MODEL_POINTER.read_text().strip()
        if saved:
            path = Path(saved)
            if path.exists():
                return path
    return default_model_path()


def format_generated_text(text: str) -> str:
    text = re.sub(r"\s+([,.:;?!%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"([.!?]\s+)", text)
    text = ""
    capitalize_next = True

    for part in sentences:
        if capitalize_next:
            part = part[:1].upper() + part[1:]
            capitalize_next = False

        text += part

        if re.search(r"[.!?]\s+$", part):
            capitalize_next = True

    sentence_list = re.split(r"(?<=[.!?])\s+", text)
    paragraphs = []

    for i in range(0, len(sentence_list), 4):
        paragraphs.append(" ".join(sentence_list[i : i + 4]))

    return "\n\n".join(paragraphs)


def load_vocab() -> tuple[dict[str, int], dict[int, str]]:
    vocab_path = ROOT / "vocab.json"
    word_path = ROOT / "word_to_int.json"
    int_path = ROOT / "int_to_word.json"

    if vocab_path.exists():
        data = json.loads(vocab_path.read_text())
        word_to_int = data["word_to_int"]
        int_to_word = {int(k): v for k, v in data["int_to_word"].items()}
        return word_to_int, int_to_word

    if word_path.exists() and int_path.exists():
        word_to_int = json.loads(word_path.read_text())
        int_to_word = {int(k): v for k, v in json.loads(int_path.read_text()).items()}
        return word_to_int, int_to_word

    for output_path in (BUNDLED_OUTPUT_PATH, DOWNLOAD_OUTPUT_PATH):
        if not output_path.exists():
            continue
        lines = output_path.read_text(errors="replace").splitlines()
        try:
            word_index = lines.index("Mapping the word to int") + 1
            int_index = lines.index("Mapping the int to word") + 1
            word_to_int = ast.literal_eval(lines[word_index])
            int_to_word = {int(k): v for k, v in ast.literal_eval(lines[int_index]).items()}
            return word_to_int, int_to_word
        except (ValueError, SyntaxError, IndexError) as exc:
            raise ValueError(f"Could not parse vocabulary from {output_path}: {exc}") from exc

    return {}, {}


def init_model() -> None:
    global MODEL, TORCH, WORD_TO_INT, INT_TO_WORD, MODEL_ERROR, ACTIVE_MODEL_PATH

    MODEL = None
    ACTIVE_MODEL_PATH = active_model_path()

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on local install
        MODEL_ERROR = f"Torch is not installed: {exc}"
        return

    WORD_TO_INT, INT_TO_WORD = load_vocab()
    if not WORD_TO_INT or not INT_TO_WORD:
        MODEL_ERROR = "Missing vocab.json or word_to_int.json/int_to_word.json."
        return

    class WordLSTM(nn.Module):
        def __init__(self, vocab_size: int, embed_size: int = 128, hidden_size: int = 128, num_layers: int = 3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, inputs, hidden):
            embedded = self.embedding(inputs)
            output, hidden = self.lstm(embedded, hidden)
            output = self.fc(output)
            return output, hidden

        def init_hidden(self, batch_size: int):
            weight = next(self.parameters())
            h = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
            c = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
            return h, c

    try:
        state = torch.load(ACTIVE_MODEL_PATH, map_location=DEVICE)
        model = WordLSTM(len(WORD_TO_INT))
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        MODEL = model
        TORCH = torch
        MODEL_ERROR = ""
    except Exception as exc:  # pragma: no cover - depends on checkpoint compatibility
        MODEL_ERROR = f"Could not load model: {exc}"


def generate(model: Any, prompt: str, top_k: int | None = None, length: int = 200, temperature: float = 1.0) -> str:
    model.eval()
    text = prompt.lower().split()
    hc = model.init_hidden(1)
    hc = tuple(h.to(DEVICE) for h in hc)

    length = length - len(text)

    for _ in range(max(0, length)):
        if len(text) <= SEQ_LEN:
            tokens = [WORD_TO_INT[w] for w in text if w in WORD_TO_INT]
        else:
            tokens = [WORD_TO_INT[w] for w in text[-SEQ_LEN:] if w in WORD_TO_INT]

        if not tokens:
            tokens = [0]

        x = TORCH.tensor([tokens], device=DEVICE)
        output, hc = model(x, hc)
        logits = output[0][-1]
        logits = logits / max(0.1, temperature)
        probs = TORCH.softmax(logits, dim=0)

        if top_k is None:
            idx = TORCH.multinomial(probs, num_samples=1).item()
        else:
            top_k = max(1, min(int(top_k), probs.numel()))
            top_probs, top_indices = TORCH.topk(probs, top_k)
            top_probs = top_probs / top_probs.sum()
            choice = TORCH.multinomial(top_probs, num_samples=1).item()
            idx = top_indices[choice].item()

        text.append(INT_TO_WORD[idx])

    return format_generated_text(" ".join(text))


def demo_sample(prompt: str, length: int) -> str:
    words = [word for word in prompt.lower().split() if word] or ["anna"]
    bank = [
        "lantern",
        "chapter",
        "house",
        "remembered",
        "softly",
        "garden",
        "window",
        "letter",
        "river",
        "evening",
        "voice",
        "secret",
        "paper",
        "almost",
        "again",
        "stood",
        "door",
        "silver",
        "quiet",
        "promise",
        "morning",
        "road",
        "watching",
        "story",
        "name",
        "world",
        "light",
        "found",
        "dream",
        "knew",
        "room",
        "held",
    ]

    while len(words) < length:
        next_word = choice(bank)
        if len(words) % 21 == 0:
            next_word += choice([".", ".", ",", "?", ";"])
        if random() < 0.03:
            next_word = "Anna"
        words.append(next_word)

    return " ".join(words)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/status":
            self.send_json(
                {
                    "ready": MODEL is not None,
                    "error": MODEL_ERROR,
                    "checkpoint": ACTIVE_MODEL_PATH.name,
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload-weights":
            self.handle_weights_upload()
            return

        if path != "/api/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            prompt = str(payload.get("prompt", "")).strip()
            requested_length = max(20, min(600, int(payload.get("length", 300))))
            temperature = max(0.1, min(2.0, float(payload.get("temperature", 0.8))))
            top_k_value = payload.get("top_k", 10)
            top_k = None if top_k_value in (None, "", "none") else max(1, min(100, int(top_k_value)))
        except Exception:
            self.send_json({"error": "Invalid request payload."}, 400)
            return

        if not prompt:
            self.send_json({"error": "Prompt is required."}, 400)
            return

        if MODEL is not None:
            text = generate(MODEL.to(DEVICE), prompt, top_k=top_k, length=requested_length, temperature=temperature)
            self.send_json({"text": text, "mode": "model"})
            return

        text = format_generated_text(demo_sample(prompt, requested_length))
        self.send_json({"text": text, "mode": "demo", "error": MODEL_ERROR})

    def handle_weights_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "Expected a multipart file upload."}, 400)
            return

        try:
            import cgi

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            field = form["weights"] if "weights" in form else None
            if field is None or not getattr(field, "filename", ""):
                self.send_json({"error": "No weights file was uploaded."}, 400)
                return

            original_name = Path(field.filename).name
            if not original_name.lower().endswith((".pth", ".pt", ".bin")):
                self.send_json({"error": "Upload a .pth, .pt, or .bin checkpoint file."}, 400)
                return

            UPLOAD_DIR.mkdir(exist_ok=True)
            destination = UPLOAD_DIR / original_name
            with destination.open("wb") as output:
                shutil.copyfileobj(field.file, output)

            ACTIVE_MODEL_POINTER.write_text(str(destination))
            init_model()

            if MODEL is not None:
                self.send_json(
                    {
                        "ready": True,
                        "message": f"Uploaded {original_name}. LSTM generation is ready.",
                        "checkpoint": original_name,
                    }
                )
                return

            self.send_json(
                {
                    "ready": False,
                    "message": f"Uploaded {original_name}. Still in demo mode: {MODEL_ERROR}",
                    "checkpoint": original_name,
                }
            )
        except Exception as exc:
            self.send_json({"error": f"Upload failed: {exc}"}, 500)


def main() -> None:
    init_model()
    host = os.environ.get("HOST", "127.0.0.1")
    env_port = os.environ.get("PORT")

    if env_port:
        port = int(env_port)
        server = ThreadingHTTPServer((host, port), Handler)
    else:
        server = None
        port = 8000
        for candidate in range(8000, 8011):
            try:
                server = ThreadingHTTPServer((host, candidate), Handler)
                port = candidate
                break
            except OSError:
                continue
        if server is None:
            raise OSError("No available local port found between 8000 and 8010.")

    print(f"Anna Karenina Word Generator running at http://{host}:{port}")
    if MODEL_ERROR:
        print(f"Demo mode: {MODEL_ERROR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
