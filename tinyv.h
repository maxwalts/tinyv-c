#ifndef TINYV_H
#define TINYV_H

#include <stdlib.h>

struct FileHeader
{
    int version;
    int num_vectors;
};

struct VectorHeader
{
    int vector_size;
    int label_len;
};

typedef struct
{
    size_t current_size;
    size_t max_size;
    float *data;
    char *label;
} Vector;

typedef struct
{
    Vector **data;
    size_t current_size;
    size_t max_size;
} VectorStore;

Vector *create_vector(size_t dim);
Vector *create_labeled_vector(size_t dim, const char *label);
void add_to_vector(Vector *v, float value);
void free_vector(Vector *v);

VectorStore *create_vectorstore(size_t initial_size);
void add_to_vectorstore(VectorStore *vs, Vector *v);
VectorStore *read_vectorstore_from_file(const char *filename);
void write_vectorstore_to_file(VectorStore *vs, const char *filename);
void free_vectorstore(VectorStore *vs);

float dot_product(Vector *v1, Vector *v2);
float cosine_similarity(Vector *v1, Vector *v2);
Vector *nearest_vector(VectorStore *vs, Vector *query);

#endif /* TINYV_H */
