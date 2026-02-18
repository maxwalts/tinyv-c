#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "tinyv.h"

// ======= Vector =======

Vector *create_vector(size_t dim)
{
    Vector *v = malloc(sizeof(Vector));
    v->data = malloc(sizeof(float) * dim);
    v->current_size = 0;
    v->max_size = dim;
    v->label = NULL;
    return v;
}

Vector *create_labeled_vector(size_t dim, const char *label)
{
    Vector *v = create_vector(dim);
    v->label = malloc(strlen(label) + 1);
    strcpy(v->label, label);
    return v;
}

static void resize_vector(Vector *v)
{
    v->max_size *= 2;
    v->data = realloc(v->data, sizeof(float) * v->max_size);
}

void add_to_vector(Vector *v, float value)
{
    if (v->current_size == v->max_size)
        resize_vector(v);
    v->data[v->current_size++] = value;
}

void free_vector(Vector *v)
{
    free(v->data);
    free(v->label);
    free(v);
}

// ======= Math =======

float dot_product(Vector *v1, Vector *v2)
{
    assert(v1->current_size == v2->current_size);
    float result = 0;
    for (size_t i = 0; i < v1->current_size; ++i)
        result += v1->data[i] * v2->data[i];
    return result;
}

float cosine_similarity(Vector *v1, Vector *v2)
{
    float dot = dot_product(v1, v2);
    float norm1 = sqrtf(dot_product(v1, v1));
    float norm2 = sqrtf(dot_product(v2, v2));
    if (norm1 == 0.0f || norm2 == 0.0f)
        return 0.0f;
    return dot / (norm1 * norm2);
}

// ======= VectorStore =======

VectorStore *create_vectorstore(size_t initial_size)
{
    VectorStore *vs = malloc(sizeof(VectorStore));
    vs->data = malloc(sizeof(Vector *) * initial_size);
    vs->current_size = 0;
    vs->max_size = initial_size;
    return vs;
}

static void resize_vectorstore(VectorStore *vs)
{
    vs->max_size *= 2;
    vs->data = realloc(vs->data, sizeof(Vector *) * vs->max_size);
}

void add_to_vectorstore(VectorStore *vs, Vector *v)
{
    if (vs->current_size == vs->max_size)
        resize_vectorstore(vs);
    vs->data[vs->current_size++] = v;
}

void free_vectorstore(VectorStore *vs)
{
    for (size_t i = 0; i < vs->current_size; ++i)
        free_vector(vs->data[i]);
    free(vs->data);
    free(vs);
}

// Returns the vector in vs with the highest cosine similarity to query.
Vector *nearest_vector(VectorStore *vs, Vector *query)
{
    assert(query != NULL);
    assert(vs->current_size > 0);

    Vector *nearest = vs->data[0];
    float best_sim = cosine_similarity(query, nearest);

    for (size_t i = 1; i < vs->current_size; ++i)
    {
        float sim = cosine_similarity(query, vs->data[i]);
        if (sim > best_sim)
        {
            nearest = vs->data[i];
            best_sim = sim;
        }
    }

    return nearest;
}

// ======= File I/O =======
// Format: FileHeader | for each vector: VectorHeader | label bytes | float32 data

VectorStore *read_vectorstore_from_file(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
        return NULL;

    struct FileHeader fh;
    fread(&fh, sizeof(fh), 1, file);

    VectorStore *vs = create_vectorstore(fh.num_vectors > 0 ? fh.num_vectors : 1);

    for (int i = 0; i < fh.num_vectors; ++i)
    {
        struct VectorHeader vh;
        fread(&vh, sizeof(vh), 1, file);

        char *label = NULL;
        if (vh.label_len > 0)
        {
            label = malloc(vh.label_len + 1);
            fread(label, 1, vh.label_len, file);
            label[vh.label_len] = '\0';
        }

        Vector *v = create_vector(vh.vector_size);
        v->current_size = vh.vector_size;
        fread(v->data, sizeof(float), vh.vector_size, file);
        v->label = label;

        add_to_vectorstore(vs, v);
    }

    fclose(file);
    return vs;
}

void write_vectorstore_to_file(VectorStore *vs, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
        return;

    struct FileHeader fh = {1, (int)vs->current_size};
    fwrite(&fh, sizeof(fh), 1, file);

    for (size_t i = 0; i < vs->current_size; ++i)
    {
        Vector *v = vs->data[i];
        int label_len = v->label ? (int)strlen(v->label) : 0;
        struct VectorHeader vh = {(int)v->current_size, label_len};
        fwrite(&vh, sizeof(vh), 1, file);
        if (label_len > 0)
            fwrite(v->label, 1, label_len, file);
        fwrite(v->data, sizeof(float), v->current_size, file);
    }

    fclose(file);
}

