#include "level_1.cuh"

void vectorAbsMinMax() {
	printf("---- Demo abs min-max element in a vector ----\n");
	const int n = 6;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	float* x = new float[n];
	generateVector(x, n, 0, 100);
	printVector(x, n, "x");
	float* d_x;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	stat = cublasCreate(&handle);
	// Copy vector to device
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	int result;
	stat = cublasIsamax(handle, n, d_x, 1, &result);
	printf("max |x[i]|: \t%.0f\n", fabs(x[result - 1]));
	stat = cublasIsamin(handle, n, d_x, 1, &result);
	printf("min |x[i]|: \t%.0f\n", fabs(x[result - 1]));
	cudaFree(d_x);
	cublasDestroy(handle);
	delete[] x;
}

void vectorAbsSum() {
	printf("---- Demo abs sum of a vector ---\n");
	const int n = 6;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	float* x = new float[n];
	generateVector(x, n, 0, 100);
	printVector(x, n, "x");
	float* d_x;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(x));
	stat = cublasCreate(&handle);
	// Copy vector to device
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	float result;
	stat = cublasSasum(handle, n, d_x, 1, &result);
	printf("sum |x[i]|: \t%.0f\n", result);
	cudaFree(d_x);
	cublasDestroy(handle);
	free(x);
}

void vectorScalar() {
	printf("---- Demo a * x + y ---\n");
}
