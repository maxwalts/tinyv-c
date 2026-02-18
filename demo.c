#include <stdio.h>
#include "tinyv.h"

// Demonstrates tinyv with hand-crafted 4D embeddings that cluster by theme.
// Dimensions loosely represent: [animal, vehicle, small, fast]
int main()
{
    float raw[][4] = {
        {0.9f, 0.1f, 0.8f, 0.6f},  // cat
        {0.8f, 0.1f, 0.7f, 0.5f},  // dog
        {0.7f, 0.1f, 0.1f, 0.2f},  // elephant
        {0.1f, 0.9f, 0.4f, 0.8f},  // car
        {0.1f, 0.8f, 0.5f, 0.6f},  // bicycle
        {0.1f, 0.9f, 0.2f, 0.5f},  // truck
    };
    const char *labels[] = {"cat", "dog", "elephant", "car", "bicycle", "truck"};
    int n = 6;

    VectorStore *vs = create_vectorstore(n);
    for (int i = 0; i < n; ++i)
    {
        Vector *v = create_labeled_vector(4, labels[i]);
        for (int j = 0; j < 4; ++j)
            add_to_vector(v, raw[i][j]);
        add_to_vectorstore(vs, v);
    }

    // Persist to disk and reload
    write_vectorstore_to_file(vs, "vectorstore.bin");
    free_vectorstore(vs);
    vs = read_vectorstore_from_file("vectorstore.bin");

    // Query: something cat-like
    Vector *query = create_labeled_vector(4, "small animal");
    float q[] = {0.85f, 0.1f, 0.75f, 0.55f};
    for (int i = 0; i < 4; ++i)
        add_to_vector(query, q[i]);

    printf("Query: \"%s\"\n", query->label);
    printf("Nearest neighbor: %s\n\n", nearest_vector(vs, query)->label);

    printf("All cosine similarities:\n");
    for (size_t i = 0; i < vs->current_size; ++i)
    {
        float sim = cosine_similarity(query, vs->data[i]);
        printf("  %-12s %.4f\n", vs->data[i]->label, sim);
    }

    free_vector(query);
    free_vectorstore(vs);
    return 0;
}
