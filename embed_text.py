"""embed_text.py – embed text with all-MiniLM-L6-v2 (ONNX) and add it to a tinyv vectorstore.

No PyTorch required — uses onnxruntime + huggingface_hub.

Setup:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install onnxruntime numpy huggingface_hub tokenizers

Usage:
    python embed_text.py "a cat sleeping" --label "cat"
    python embed_text.py "a car on the highway" --label "car" --store my.bin

The model (~23 MB) is downloaded automatically on first run.
The vectorstore file is created if it does not exist.
"""

import argparse
import struct
import os
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TINYV_VERSION = 1
MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_REPO = "optimum-internal-testing/tiny-random-bert"  # fallback placeholder


def _get_model():
    """Download and cache the ONNX model + tokenizer. Returns (session, tokenizer)."""
    import onnxruntime as ort

    # Use the ONNX export of all-MiniLM-L6-v2 from optimum-models
    onnx_path = hf_hub_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        filename="onnx/model.onnx",
    )
    tokenizer_path = hf_hub_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        filename="tokenizer.json",
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=128)
    return session, tokenizer


def embed_text(text):
    """Returns a float32 numpy array of shape (384,)."""
    session, tokenizer = _get_model()

    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    # outputs[0] is token embeddings (1, seq_len, 384) — mean-pool over tokens
    token_embeddings = outputs[0]
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    embedding = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)
    embedding = embedding.squeeze()

    # L2-normalize so cosine similarity == dot product
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding.astype(np.float32)


def read_vectorstore(path):
    """Returns (version, list of (label, floats))."""
    if not os.path.exists(path):
        return TINYV_VERSION, []
    vectors = []
    with open(path, "rb") as f:
        version, num_vectors = struct.unpack("ii", f.read(8))
        for _ in range(num_vectors):
            vector_size, label_len = struct.unpack("ii", f.read(8))
            label = f.read(label_len).decode("utf-8") if label_len > 0 else ""
            floats = struct.unpack(f"{vector_size}f", f.read(4 * vector_size))
            vectors.append((label, list(floats)))
    return version, vectors


def write_vectorstore(path, version, vectors):
    with open(path, "wb") as f:
        f.write(struct.pack("ii", version, len(vectors)))
        for label, floats in vectors:
            label_bytes = label.encode("utf-8")
            f.write(struct.pack("ii", len(floats), len(label_bytes)))
            f.write(label_bytes)
            f.write(struct.pack(f"{len(floats)}f", *floats))


def main():
    parser = argparse.ArgumentParser(description="Embed text and add to a tinyv vectorstore")
    parser.add_argument("text", help="Text to embed")
    parser.add_argument("--store", default="vectorstore.bin", help="Path to vectorstore file")
    parser.add_argument("--label", help="Label for this vector (defaults to input text)")
    args = parser.parse_args()

    label = args.label or args.text
    print(f'Embedding: "{args.text}"')
    embedding = embed_text(args.text)
    print(f"Embedding dim: {len(embedding)}")

    version, vectors = read_vectorstore(args.store)
    vectors.append((label, embedding.tolist()))
    write_vectorstore(args.store, version, vectors)

    print(f"Saved to {args.store} ({len(vectors)} vector(s) total)")


if __name__ == "__main__":
    main()
