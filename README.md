# tinyv-c

A minimal nearest-neighbor embedding database written in C (~250 lines).

```
$ gcc -o demo demo.c tinyv.c -lm && ./demo

Query: "small animal"
Nearest neighbor: cat

All cosine similarities:
  cat          0.9999
  dog          0.9999
  elephant     0.8428
  car          0.5688
  bicycle      0.6132
  truck        0.4506
```

## Features

- Cosine similarity search over float vectors
- Labeled vectors with string metadata
- Persistent binary storage — save and reload vectorstores from disk
- Dynamic resizing — no fixed capacity limit
- Single header + source file, no dependencies beyond libc and libm

## Files

| File | Description |
|---|---|
| `tinyv.h` | Public API |
| `tinyv.c` | Library implementation |
| `demo.c` | Demo with hand-crafted 4D embeddings |
| `query.c` | CLI tool for querying a `.bin` vectorstore |
| `embed_text.py` | Embed text with all-MiniLM-L6-v2 (ONNX) and save to a vectorstore |

## Build & run (demo)

```bash
gcc -o demo demo.c tinyv.c -lm
./demo
```

## Real text embeddings

`embed_text.py` uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(384-dim, ~23 MB ONNX model, no PyTorch required).

**Setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install onnxruntime numpy huggingface_hub tokenizers
```

**Build the vectorstore and query from C:**
```bash
# Add entries
python embed_text.py "a cat sleeping"       --label "cat"   --store animals.bin
python embed_text.py "a dog playing fetch"  --label "dog"   --store animals.bin
python embed_text.py "a car on the highway" --label "car"   --store animals.bin

# Append a query vector, then search with C
python embed_text.py "a kitten napping" --label "query: kitten" --store animals.bin
gcc -o query query.c tinyv.c -lm
./query animals.bin
```

```
Query: "query: kitten"
Nearest neighbor: cat

All cosine similarities:
  cat          0.7159
  dog          0.1693
  car          0.1329
```

## API

```c
// Vectors
Vector *v = create_labeled_vector(384, "my text");
add_to_vector(v, 0.12f);  // add each dimension

// VectorStore
VectorStore *vs = create_vectorstore(16);
add_to_vectorstore(vs, v);

// Search
Vector *result = nearest_vector(vs, query);
printf("Nearest: %s\n", result->label);

float sim = cosine_similarity(v1, v2);

// Persist
write_vectorstore_to_file(vs, "store.bin");
VectorStore *loaded = read_vectorstore_from_file("store.bin");

// Free
free_vectorstore(vs);
```

## Binary file format

Each `.bin` file is a flat binary with the following layout:

| Field | Type | Description |
|---|---|---|
| `version` | int32 | Format version (1) |
| `num_vectors` | int32 | Number of vectors |
| — repeated per vector — | | |
| `vector_size` | int32 | Number of floats |
| `label_len` | int32 | Label byte length |
| `label` | bytes | UTF-8 label string |
| `data` | float32[] | Vector values |
