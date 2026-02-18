#include <stdio.h>
#include <stdlib.h>
#include "tinyv.h"

// Query a tinyv vectorstore from the command line.
// The query vector must already be in the store as the last entry
// (append it with embed_text.py before running).
//
// Usage:
//   gcc -o query query.c tinyv.c -lm
//   python embed_text.py "a kitten napping" --label "query" --store store.bin
//   ./query store.bin
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: query <store.bin>\n");
        return 1;
    }

    VectorStore *vs = read_vectorstore_from_file(argv[1]);
    if (!vs)
    {
        fprintf(stderr, "error: could not open %s\n", argv[1]);
        return 1;
    }
    if (vs->current_size < 2)
    {
        fprintf(stderr, "error: store needs at least 2 vectors (corpus + query)\n");
        free_vectorstore(vs);
        return 1;
    }

    // Last vector in the store is treated as the query
    Vector *query = vs->data[vs->current_size - 1];
    vs->current_size -= 1;

    printf("Query: \"%s\"\n", query->label);
    printf("Nearest neighbor: %s\n\n", nearest_vector(vs, query)->label);

    printf("All cosine similarities:\n");
    for (size_t i = 0; i < vs->current_size; ++i)
    {
        float sim = cosine_similarity(query, vs->data[i]);
        printf("  %-20s %.4f\n", vs->data[i]->label, sim);
    }

    vs->current_size += 1;  // restore so free_vectorstore cleans up the query too
    free_vectorstore(vs);
    return 0;
}
