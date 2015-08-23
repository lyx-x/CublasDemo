#include <cstdio>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "helper.h"

// Find absolute maximum and minimum element (or index) in a vector
void vectorAbsMinMax();

// Sum up elements of a vector
void vectorAbsSum();

// Calculate a * x + y
void vectorScalar();

// Copy the vector x to y
void vectorCopyOnDevice();

// Calculate the scalar product between two vectors
void vectorDotProduct();
