from __future__ import annotations

import json
import os
import re
import shutil
import ast
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from random import choice, random
from typing import Any, TypedDict
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
BUNDLED_MODEL_PATH = ROOT / "models" / "LSTM_Annie.pth"
DOWNLOAD_MODEL_PATH = Path("/Users/michaelwu/Downloads/LSTM_Annie.pth")
HF_MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "CatFatOw123/Anna_Karenina_Model")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "LSTM_Annie.pth")
WAR_BUNDLED_MODEL_PATH = ROOT / "models" / "war_lstm.pth"
WAR_DOWNLOAD_MODEL_PATH = Path("/Users/michaelwu/Downloads/war_lstm.pth")
WAR_HF_MODEL_REPO_ID = os.environ.get("WAR_HF_MODEL_REPO_ID", "CatFatOw123/War_Of_Worlds_Model")
WAR_HF_MODEL_FILENAME = os.environ.get("WAR_HF_MODEL_FILENAME", "war_lstm.pth")
BUNDLED_OUTPUT_PATH = ROOT / "output.txt"
DOWNLOAD_OUTPUT_PATH = Path("/Users/michaelwu/Downloads/output.txt")
WAR_MAPPINGS_PATH = Path("/Users/michaelwu/Downloads/mappings.txt")
UPLOAD_DIR = ROOT / "uploaded_weights"
ACTIVE_MODEL_POINTER = UPLOAD_DIR / "active_checkpoint.txt"
SEQ_LEN = 100

TORCH: Any = None
DEVICE = "cpu"


class BookState(TypedDict):
    key: str
    title: str
    model: Any
    word_to_int: dict[str, int]
    int_to_word: dict[int, str]
    error: str
    active_model_path: Path


BOOKS: dict[str, dict[str, Any]] = {
    "anna": {
        "title": "Anna Karenina",
        "model_paths": (BUNDLED_MODEL_PATH, DOWNLOAD_MODEL_PATH),
        "hf_repo_id": HF_MODEL_REPO_ID,
        "hf_filename": HF_MODEL_FILENAME,
        "vocab_path": ROOT / "vocab.json",
        "word_path": ROOT / "word_to_int.json",
        "int_path": ROOT / "int_to_word.json",
        "output_paths": (BUNDLED_OUTPUT_PATH, DOWNLOAD_OUTPUT_PATH),
        "demo_bank": [
            "princess",
            "vronsky",
            "levin",
            "drawing",
            "room",
            "smile",
            "carriage",
            "letter",
            "heart",
            "dolly",
            "anna",
            "prince",
            "silence",
            "love",
            "moscow",
        ],
        "demo_name": "Anna",
    },
    "war": {
        "title": "The War of the Worlds",
        "model_paths": (WAR_BUNDLED_MODEL_PATH, WAR_DOWNLOAD_MODEL_PATH),
        "hf_repo_id": WAR_HF_MODEL_REPO_ID,
        "hf_filename": WAR_HF_MODEL_FILENAME,
        "vocab_path": ROOT / "war_vocab.json",
        "word_path": ROOT / "war_word_to_int.json",
        "int_path": ROOT / "war_int_to_word.json",
        "output_paths": (WAR_MAPPINGS_PATH,),
        "demo_bank": [
            "martians",
            "cylinder",
            "woking",
            "heat",
            "ray",
            "smoke",
            "tripod",
            "london",
            "earth",
            "pit",
            "red",
            "weed",
            "destruction",
            "death",
            "darkness",
        ],
        "demo_name": "Martians",
    },
}
BOOK_STATES: dict[str, BookState] = {}


def default_model_path(config: dict[str, Any]) -> Path:
    for path in config["model_paths"]:
        if path.exists():
            return path

    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=config["hf_repo_id"],
                filename=config["hf_filename"],
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not find a local model file or download {config['hf_repo_id']}/{config['hf_filename']}: {exc}"
        ) from exc


def active_model_path(book: str, config: dict[str, Any]) -> Path:
    pointer = UPLOAD_DIR / f"{book}_active_checkpoint.txt"
    if pointer.exists():
        saved = pointer.read_text().strip()
        if saved:
            path = Path(saved)
            if path.exists():
                return path
    return default_model_path(config)


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


def load_vocab(config: dict[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
    if config["vocab_path"].exists():
        data = json.loads(config["vocab_path"].read_text())
        word_to_int = data["word_to_int"]
        int_to_word = {int(k): v for k, v in data["int_to_word"].items()}
        return word_to_int, int_to_word

    if config["word_path"].exists() and config["int_path"].exists():
        word_to_int = json.loads(config["word_path"].read_text())
        int_to_word = {int(k): v for k, v in json.loads(config["int_path"].read_text()).items()}
        return word_to_int, int_to_word

    for output_path in config["output_paths"]:
        if not output_path.exists():
            continue
        lines = output_path.read_text(errors="replace").splitlines()
        try:
            if "Mapping the word to int" in lines:
                word_index = lines.index("Mapping the word to int") + 1
                int_index = lines.index("Mapping the int to word") + 1
            else:
                word_index = 0
                int_index = 2
            word_to_int = ast.literal_eval(lines[word_index])
            int_to_word = {int(k): v for k, v in ast.literal_eval(lines[int_index]).items()}
            return word_to_int, int_to_word
        except (ValueError, SyntaxError, IndexError) as exc:
            raise ValueError(f"Could not parse vocabulary from {output_path}: {exc}") from exc

    return {}, {}


def build_model_class(torch_module: Any):
    import torch.nn as nn

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

    return WordLSTM


def init_model(book: str, config: dict[str, Any]) -> BookState:
    global TORCH

    state: BookState = {
        "key": book,
        "title": config["title"],
        "model": None,
        "word_to_int": {},
        "int_to_word": {},
        "error": "",
        "active_model_path": Path(config["hf_filename"]),
    }
    try:
        state["active_model_path"] = active_model_path(book, config)
    except Exception as exc:
        state["error"] = f"Could not locate model checkpoint: {exc}"
        return state

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local install
        state["error"] = f"Torch is not installed: {exc}"
        return state

    word_to_int, int_to_word = load_vocab(config)
    if not word_to_int or not int_to_word:
        state["error"] = f"Missing vocabulary for {config['title']}."
        return state

    try:
        checkpoint = torch.load(state["active_model_path"], map_location=DEVICE)
        model = build_model_class(torch)(len(word_to_int))
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        TORCH = torch
        state["model"] = model
        state["word_to_int"] = word_to_int
        state["int_to_word"] = int_to_word
    except Exception as exc:  # pragma: no cover - depends on checkpoint compatibility
        state["error"] = f"Could not load model: {exc}"
    return state


def init_models() -> None:
    BOOK_STATES.clear()
    for key, config in BOOKS.items():
        BOOK_STATES[key] = init_model(key, config)


def generate(
    model: Any,
    word_to_int: dict[str, int],
    int_to_word: dict[int, str],
    prompt: str,
    top_k: int | None = None,
    length: int = 200,
    temperature: float = 1.0,
) -> str:
    model.eval()
    text = prompt.lower().split()
    hc = model.init_hidden(1)
    hc = tuple(h.to(DEVICE) for h in hc)

    length = length - len(text)

    for _ in range(max(0, length)):
        if len(text) <= SEQ_LEN:
            tokens = [word_to_int[w] for w in text if w in word_to_int]
        else:
            tokens = [word_to_int[w] for w in text[-SEQ_LEN:] if w in word_to_int]

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

        text.append(int_to_word[idx])

    return format_generated_text(" ".join(text))


def demo_sample(prompt: str, length: int, config: dict[str, Any]) -> str:
    words = [word for word in prompt.lower().split() if word] or [config["demo_name"].lower()]
    bank = config["demo_bank"]

    while len(words) < length:
        next_word = choice(bank)
        if len(words) % 21 == 0:
            next_word += choice([".", ".", ",", "?", ";"])
        if random() < 0.03:
            next_word = config["demo_name"]
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
            books = {}
            for key, state in BOOK_STATES.items():
                books[key] = {
                    "title": state["title"],
                    "ready": state["model"] is not None,
                    "error": state["error"],
                    "checkpoint": state["active_model_path"].name,
                    "vocab_size": len(state["word_to_int"]),
                }
            self.send_json(
                {
                    "ready": any(state["model"] is not None for state in BOOK_STATES.values()),
                    "books": books,
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
            temperature = max(0.1, min(2.0, float(payload.get("temperature", 0.3))))
            top_k_value = payload.get("top_k", 3)
            top_k = None if top_k_value in (None, "", "none") else max(1, min(100, int(top_k_value)))
            book = str(payload.get("book", "anna")).strip().lower()
        except Exception:
            self.send_json({"error": "Invalid request payload."}, 400)
            return

        if book not in BOOKS:
            self.send_json({"error": "Unknown book selection."}, 400)
            return

        if not prompt:
            self.send_json({"error": "Prompt is required."}, 400)
            return

        state = BOOK_STATES[book]
        config = BOOKS[book]
        if state["model"] is not None:
            text = generate(
                state["model"].to(DEVICE),
                state["word_to_int"],
                state["int_to_word"],
                prompt,
                top_k=top_k,
                length=requested_length,
                temperature=temperature,
            )
            self.send_json({"text": text, "mode": "model", "book": book})
            return

        text = format_generated_text(demo_sample(prompt, requested_length, config))
        self.send_json({"text": text, "mode": "demo", "book": book, "error": state["error"]})

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
            book_field = form["book"] if "book" in form else None
            book = "anna"
            if book_field is not None and getattr(book_field, "value", ""):
                book = str(book_field.value).strip().lower()
            if book not in BOOKS:
                self.send_json({"error": "Unknown book selection."}, 400)
                return

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

            pointer = UPLOAD_DIR / f"{book}_active_checkpoint.txt"
            pointer.write_text(str(destination))
            BOOK_STATES[book] = init_model(book, BOOKS[book])
            state = BOOK_STATES[book]

            if state["model"] is not None:
                self.send_json(
                    {
                        "ready": True,
                        "message": f"Uploaded {original_name}. {state['title']} generation is ready.",
                        "checkpoint": original_name,
                        "book": book,
                    }
                )
                return

            self.send_json(
                {
                    "ready": False,
                    "message": f"Uploaded {original_name}. Still in demo mode: {state['error']}",
                    "checkpoint": original_name,
                    "book": book,
                }
            )
        except Exception as exc:
            self.send_json({"error": f"Upload failed: {exc}"}, 500)


def main() -> None:
    init_models()
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

    print(f"Book Generator running at http://{host}:{port}")
    for state in BOOK_STATES.values():
        if state["error"]:
            print(f"{state['title']} demo mode: {state['error']}")
        else:
            print(f"{state['title']} ready: {state['active_model_path'].name}")
    server.serve_forever()


if __name__ == "__main__":
    main()
