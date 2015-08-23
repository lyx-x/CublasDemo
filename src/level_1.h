#include <cstdio>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "helper.h"

// Find absolute maximum and minimum element (or index) in a vector
void vectorAbsMinMax();

// Sum up elements of a vector
void vectorAbsSum();

// Calculate a * x
void vectorScalar();

// Calculate a * x + y
void vectorScalarPlus();

// Copy the vector x to y
void vectorCopyOnDevice();

// Calculate the scalar product between two vectors
void vectorDotProduct();

// Return the Euclidean norm of the vector
void vectorNorm();

// Swap two vectors
void vectorSwap();
