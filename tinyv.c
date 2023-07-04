#include <stdio.h>
#include <assert.h>
#include "tinyv.h"
// Needed to use Python/C API
#define PY_SSIZE_T_CLEAN
#include <Python.h> // TODO: add path to Python.h

// ======= Vector =======
Vector *create_vector(size_t initial_size)
{
    Vector *v = malloc(sizeof(Vector));
    v->data = malloc(sizeof(int) * initial_size);
    v->current_size = 0;
    v->max_size = initial_size;
    return v;
}

void resize_vector(Vector *v)
{
    v->max_size *= 2;
    v->data = realloc(v->data, sizeof(int) * v->max_size);
}

void add_to_vector(Vector *v, int value)
{
    if (v->current_size == v->max_size)
    {
        resize_vector(v);
    }
    v->data[v->current_size++] = value;
}

// Frees the array of integers and the vector itself
void free_vector(Vector *v)
{
    free(v->data);
    free(v);
}

float dot_product(Vector *v1, Vector *v2)
{
    assert(v1->current_size == v2->current_size);
    float result = 0;
    for (size_t i = 0; i < v1->current_size; ++i)
    {
        result += v1->data[i] * v2->data[i];
    }
    return result;
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

void resize_vectorstore(VectorStore *vs)
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

// Reads a file header, followed by a vector header and vector data for each vector in the file.
VectorStore *read_vectorstore_from_file(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
        return NULL;

    struct FileHeader fh;
    fread(&fh, sizeof(fh), 1, file);

    VectorStore *vs = create_vectorstore(fh.num_vectors);

    for (int i = 0; i < fh.num_vectors; ++i)
    {
        struct VectorHeader vh;
        fread(&vh, sizeof(vh), 1, file);

        Vector *v = create_vector(vh.vector_size);
        v->current_size = vh.vector_size;
        fread(v->data, sizeof(int), vh.vector_size, file);

        add_to_vectorstore(vs, v);
    }

    fclose(file);
    return vs;
}

// Writes a file header, followed by a vector header and vector data for each vector in the vector store.
void write_vectorstore_to_file(VectorStore *vs, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
        return;

    struct FileHeader fh = {1, vs->current_size};
    fwrite(&fh, sizeof(fh), 1, file);

    for (size_t i = 0; i < vs->current_size; ++i)
    {
        Vector *v = vs->data[i];
        struct VectorHeader vh = {v->current_size};
        fwrite(&vh, sizeof(vh), 1, file);
        fwrite(v->data, sizeof(int), v->current_size, file);
    }

    fclose(file);
}

// Frees each vector, the array of vector pointers, and the vectorstore itself.
void free_vectorstore(VectorStore *vs)
{
    for (size_t i = 0; i < vs->current_size; ++i)
        free_vector(vs->data[i]);
    free(vs->data);
    free(vs);
}

Vector *nearest_vector(VectorStore *vs, Vector *query)
{
    assert(query != NULL);
    assert(vs->current_size > 0);

    Vector *nearest = vs->data[0];
    float nearest_dist = dot_product(query, nearest);

    for (size_t i = 1; i < vs->current_size; ++i)
    {
        float dist = dot_product(query, vs->data[i]);
        if (dist < nearest_dist)
        {
            nearest = vs->data[i];
            nearest_dist = dist;
        }
    }

    return nearest;
}

// ======= Example usage with tests =======
int main()
{
    // Create a vector store with initial_size = 1
    VectorStore *vs = create_vectorstore(1);
    assert(vs->current_size == 0);

    // Create a vector with initial_size = 1
    Vector *v = create_vector(1);
    assert(v->current_size == 0);
    assert(v->max_size == 1);

    // Add some numbers to the vector
    add_to_vector(v, 1);
    add_to_vector(v, 2);
    add_to_vector(v, 3);
    assert(v->current_size == 3);
    assert(v->data[2] == 3);

    // Add the vector to the vector store
    add_to_vectorstore(vs, v);
    assert(vs->current_size == 1);
    assert(vs->data[0] == v);

    // Save the vector store to a file, then load it back into a new vector store
    write_vectorstore_to_file(vs, "vectorstore.bin");
    VectorStore *loaded_vs = read_vectorstore_from_file("vectorstore.bin");
    assert(loaded_vs->current_size == vs->current_size);
    assert(loaded_vs->data[0]->current_size == vs->data[0]->current_size);
    assert(loaded_vs->data[0]->data[0] == vs->data[0]->data[0]);

    // Free the memory for 1st vector store
    free_vectorstore(vs);

    // Print all numbers from all vectors in the loaded vector store
    for (size_t i = 0; i < loaded_vs->current_size; ++i)
    {
        Vector *v = loaded_vs->data[i];
        for (size_t j = 0; j < v->current_size; ++j)
        {
            printf("%d\n", v->data[j]);
        }
    }

    // Free the memory for the loaded vector store
    free_vectorstore(loaded_vs);

    return 0;
}
