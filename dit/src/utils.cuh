#include "dit.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

template <class T>
__host__ __device__ T ceil_div(T dividend, T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

inline int max_int(int a, int b)
{
    return a > b ? a : b;
}

inline int min_int(int a, int b)
{
    return a < b ? a : b;
}

// CUDA error checking
inline void cuda_check(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
inline void cublas_check(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status)                         \
    {                                               \
        cublas_check((status), __FILE__, __LINE__); \
    }

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck
// copied from llm.c

FILE *fopen_check(const char *path, const char *mode, const char *file, int line)
{
    FILE *fp = fopen(path, mode);
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    return fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line)
{
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb)
    {
        if (feof(stream))
        {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        }
        else if (ferror(stream))
        {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        }
        else
        {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

void fclose_check(FILE *fp, const char *file, int line)
{
    if (fclose(fp) != 0)
    {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

void fseek_check(FILE *fp, long off, int whence, const char *file, int line)
{
    if (fseek(fp, off, whence) != 0)
    {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

void *malloc_check(size_t size, const char *file, int line)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// sin/cos positional embedding lookup table

// sizeof(float) * 1024 * 1024 * 2 = 8 MB
constexpr size_t TABLE_SIZE = 1024 * 1024;

// Generate a table for the sine and cosine of 2*PI*x for x in [0,1]
// at increments of 1/TABLE_SIZE. Since the output of sin/cos is in [-1,1],
// we can use int16_t for a more compact representation, and cast to float when needed
void generate_sincos_table(int16_t *table)
{
    const float factor = (float)(2.0 * CUDART_PI_F / TABLE_SIZE);
    for (size_t i = 0; i < TABLE_SIZE; ++i)
    {
        float x = i * factor;
        table[2 * i + 0] = __float2half_rn(sinf(x));
        table[2 * i + 1] = __float2half_rn(cosf(x));
    }
}

// Given an array x of floats in [0, 1], this function will return
// a Packed128 containing the sine and cosine values for 2*PI*x.
template <typename T>
__device__ Packed128<T> sincos2pix(const T x, const int16_t *table)
{
    const int i = (int)(x * TABLE_SIZE) & (TABLE_SIZE - 1); // modulo with bitwise AND for efficiency
    Packed128<T> result;
    result[0] = __half2float(table[2 * i + 0]);
    result[1] = __half2float(table[2 * i + 1]);
    return result;
}