# tinyv-c (wip)
A tiny nearest-neighbor embedding database written in C.
- under 300 lines
- vector stores written/read locally as binary files
- includes memory management
- pytorch used for text embedding (wip)

## Example use and testing
```c
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
```
Run this code by cloning the repo and running
```bash
$ gcc tinyv.c -o tinyv
$ ./tinyv
```

