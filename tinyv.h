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
};

typedef struct
{
    size_t current_size;
    size_t max_size;
    float *data;
} Vector;

typedef struct
{
    Vector **data;
    size_t current_size;
    size_t max_size;
} VectorStore;

Vector *create_vector(size_t initial_size);
void resize_vector(Vector *v);
void add_to_vector(Vector *v, int value);
void free_vector(Vector *v);

VectorStore *create_vectorstore(size_t initial_size);
void resize_vectorstore(VectorStore *vs);
void add_to_vectorstore(VectorStore *vs, Vector *v);

VectorStore *read_vectorstore_from_file(const char *filename);
void write_vectorstore_to_file(VectorStore *vs, const char *filename);
void free_vectorstore(VectorStore *vs);

float dot_product(Vector *v1, Vector *v2);

#endif /* TINYV_H */
