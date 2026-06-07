# Anna Karenina Word Generator

Access the website here: https://catfatow.github.io/Anna-Karenina-Text-Generation/

A local web app for generating book-style passages inspired by _Anna Karenina_.
The site asks for a prompt, then produces a formatted manuscript-style passage
using either the trained LSTM backend or a built-in demo generator when model
dependencies are not available.

The app describes the model as an LSTM trained for 15 epochs on an A100 GPU to
replicate the language and cadence of _Anna Karenina_.

## Features

- Prompt-based text generation
- Word-count and temperature controls
- Manuscript-style output formatting
- Copy-to-clipboard support
- Optional upload for custom `.pth`, `.pt`, or `.bin` checkpoint weights
- Demo mode fallback when Torch or vocabulary files are missing

## Access the website

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

The checkpoint at `/Users/michaelwu/Downloads/LSTM_Annie.pth` contains a three-layer LSTM state dict with a 13,000-word vocabulary, 128-dimensional embeddings, and a 128-unit hidden size.

You can upload a different `.pth`, `.pt`, or `.bin` checkpoint from the website.
Uploaded checkpoints are stored locally in `uploaded_weights/`, which is ignored
by Git so large model files are not pushed to GitHub.

To use real model generation, install `torch` and `numpy`, then add one of these
vocabulary formats to this folder:

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

Save that as `vocab.json`, or save the two mappings separately as `word_to_int.json` and `int_to_word.json`.

Without those files, the app runs in demo mode so the website remains usable.
