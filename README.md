# Anna Karenina Word Generator

A small local website for generating book-style passages from a prompt.

The model description shown in the app notes that the generator uses an LSTM trained for 15 epochs on an A100 GPU to replicate the language and cadence of _Anna Karenina_.

## Run

```bash
python3 server.py
```

Then open `http://127.0.0.1:8000`.

## Model mode

The checkpoint at `/Users/michaelwu/Downloads/LSTM_Annie.pth` contains a three-layer LSTM state dict with a 13,000-word vocabulary, 128-dimensional embeddings, and a 128-unit hidden size.

You can upload a different `.pth`, `.pt`, or `.bin` checkpoint from the website. Uploaded checkpoints are stored locally in `uploaded_weights/`, which is ignored by Git so large model files are not pushed to GitHub.

To use real model generation, install `torch` and `numpy`, then add one of these vocabulary formats to this folder:

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
