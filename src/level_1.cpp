#include "level_1.h"

void vectorAbsMinMax() {
	printf("---- Demo ans := min|max(|x[i]|) ----\n");
	const int n = 6;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	float* x = new float[n];
	generateVector(x, n);
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
	printf("---- Demo ans := sum(|x[i]|) ----\n");
	const int n = 6;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	float* x = new float[n];
	generateVector(x, n);
	printVector(x, n, "x");
	float* d_x;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	stat = cublasCreate(&handle);
	// Copy vector to device
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	float result;
	stat = cublasSasum(handle, n, d_x, 1, &result);
	printNumber(&result, "sum |x[i]|");
	cudaFree(d_x);
	cublasDestroy(handle);
	free(x);
}

void vectorScalar() {
	printf("---- Demo ans := a * x ----\n");
	const int n = 6;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	float a = 2.0;
	printf("a:\t%.2f\n", a);
	float *x = new float[n];
	generateVector(x, n);
	printVector(x, n, "x");
	float *d_x;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	stat = cublasSscal(handle, n, &a, d_x, 1);
	stat = cublasGetVector(n, sizeof(*x), d_x, 1, x, 1);
	printVector(x, n, "ans");
	cudaFree(d_x);
	cublasDestroy(handle);
	delete[] x;
}

void vectorScalarPlus() {
	printf("---- Demo ans := a * x + y ----\n");
	const int n = 6;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	float a = 2.0;
	printf("a:\t%.2f\n", a);
	float *x = new float[n];
	generateVector(x, n);
	printVector(x, n, "x");
	float *y = new float[n];
	generateVector(y, n, -10, 10);
	printVector(y, n, "y");
	float *d_x;
	float *d_y;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	cudaStat = cudaMalloc((void**)&d_y, n * sizeof(*y));
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	stat = cublasSetVector(n, sizeof(*y), y, 1, d_y, 1);
	stat = cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1);
	stat = cublasGetVector(n, sizeof(*y), d_y, 1, y, 1);
	printVector(y, n, "ans");
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
	delete[] x;
	delete[] y;
}

void vectorCopyOnDevice() {
	printf("---- Demo y := x on device ----\n");
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	const int n = 6;
	float *x = new float[n];
	float *y = new float[n];
	generateVector(x, n);
	printVector(x, n, "x");
	float *d_x;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	float *d_y;
	cudaStat = cudaMalloc((void**)&d_y, n * sizeof(*y));
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	stat = cublasScopy(handle, n, d_x, 1, d_y, 1);
	stat = cublasGetVector(n, sizeof(*y), d_y, 1, y, 1);
	printVector(y, n, "y");
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
	delete[] x;
	delete[] y;
}

void vectorDotProduct() {
	printf("---- Demo ans := x .* y ----\n");
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	const int n = 6;
	float *x = new float[n];
	generateVector(x, n);
	printVector(x, n, "x");
	float *y = new float[n];
	generateVector(y, n);
	printVector(y, n, "y");
	float *d_x;
	float *d_y;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	cudaStat = cudaMalloc((void**)&d_y, n * sizeof(*y));
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	stat = cublasSetVector(n, sizeof(*y), y, 1, d_y, 1);
	float result;
	stat = cublasSdot(handle, n, d_x, 1, d_y, 1, &result);
	printNumber(&result, "ans");
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
	delete[] x;
	delete[] y;
}

void vectorNorm() {
	printf("---- Demo ans := ||x||_2----\n");
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	const int n = 6;
	float *x = new float[n];
	generateVector(x, n);
	printVector(x, n, "x");
	float *d_x;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	float result;
	stat = cublasSnrm2(handle, n, d_x, 1, &result);
	printNumber(&result, "ans");
	cudaFree(d_x);
	cublasDestroy(handle);
	delete[] x;
}

void vectorSwap() {
	printf("---- Demo swap(x, y) ----\n");
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	const int n = 6;
	float *x = new float[n];
	generateVector(x, n);
	printVector(x, n, "x0");
	float *y = new float[n];
	generateVector(y, n);
	printVector(y, n, "y0");
	float *d_x;
	float *d_y;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(*x));
	cudaStat = cudaMalloc((void**)&d_y, n * sizeof(*y));
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
	stat = cublasSetVector(n, sizeof(*y), y, 1, d_y, 1);
	stat = cublasSswap(handle, n, d_x, 1, d_y, 1);
	stat = cublasGetVector(n, sizeof(*x), d_x, 1, x, 1);
	stat = cublasGetVector(n, sizeof(*y), d_y, 1, y, 1);
	printVector(x, n, "x1");
	printVector(y, n, "y1");
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
	delete[] x;
	delete[] y;
}
