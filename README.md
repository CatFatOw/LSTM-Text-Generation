# Anna Karenina Word Generator

Code repository: https://github.com/CatFatOw/Anna-Karenina-Text-Generation

Note: the Python/Torch generator must run on a Python server. A static GitHub
Pages URL can show frontend files, but it cannot run the LSTM model by itself.

A local web app for generating book-style passages inspired by _Anna Karenina_.
The site asks for a prompt, then produces a formatted manuscript-style passage
using either the trained LSTM backend or a built-in demo generator when model
dependencies are not available.

The app describes the model as an LSTM trained for 15 epochs on an A100 GPU to
replicate the language and cadence of _Anna Karenina_.

The default generation settings mirror this call:

```python
print(generate(model.to(device), "Anna and the prince", top_k=10, length=300, temperature=0.8))
```

## Features

- Prompt-based text generation
- Word-count, temperature, and Top-K controls
- Manuscript-style output formatting
- Copy-to-clipboard support
- Optional upload for custom `.pth`, `.pt`, or `.bin` checkpoint weights
- Demo mode fallback when Torch or vocabulary files are missing

## Access the website

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the local server:

```bash
python3 server.py
```

Then open the website in your browser:

```text
http://127.0.0.1:8000
```

If port `8000` is already in use, the server automatically tries the next
available port from `8001` through `8010` and prints the URL in the terminal.

## Model mode

The checkpoint at `models/LSTM_Annie.pth` contains a three-layer LSTM state dict
with a 13,000-word vocabulary, 128-dimensional embeddings, and a 128-unit hidden
size. If that bundled model is missing, the server also checks
`/Users/michaelwu/Downloads/LSTM_Annie.pth` for local development.

You can upload a different `.pth`, `.pt`, or `.bin` checkpoint from the website.
Uploaded checkpoints are stored locally in `uploaded_weights/`, which is ignored
by Git so large model files are not pushed to GitHub.

Real generation needs `torch`, `numpy`, the checkpoint, and a vocabulary mapping.
The repo includes `requirements.txt` for installing Torch/Numpy and can load
`vocab.json` directly. The expected vocabulary format is:

```json
{
  "word_to_int": {
    "anna": 42
  },
  "int_to_word": {
    "42": "anna"
  }
}
```

Save that as `vocab.json`, or save the two mappings separately as
`word_to_int.json` and `int_to_word.json`.

If those JSON files are missing, the server also tries to parse the printed
training output at `/Users/michaelwu/Downloads/output.txt`, as long as it
contains the `Mapping the word to int` and `Mapping the int to word` sections.

Without those files, the app runs in demo mode so the website remains usable.

On this machine, if Anaconda Python does not have Torch, use the Torch-enabled
Python:

```bash
/usr/local/bin/python3 server.py
```
