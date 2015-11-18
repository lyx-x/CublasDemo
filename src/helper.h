#ifndef HELPER
#define HELPER

#include <cstdio>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#define fatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define callCuda(status) do {                                  		   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
    	_error << "Cuda failure: " << status;                          \
    	fatalError(_error.str());                                      \
    }                                                                  \
} while(0)

void printNumber(float *a, std::string name);

void printVector(float *x, int n, std::string name);

void generateVector(float *v, int n, int lower = 0, int upper = 100);

void printMatrix(float *x, int n, int r, int c, std::string name);

void printGpuMatrix(float* d_x, int n, int r, int c, std::string name);

void printMatrix(int *x, int n, int r, int c, std::string name);

void printGpuMatrix(int* d_x, int n, int r, int c, std::string name);


#endif
