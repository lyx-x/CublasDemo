#include "helper.h"

void printNumber(float *a, std::string name) {
	printf("%s: \t%.2f\n", name.c_str(), *a);
}

void printVector(float *x, int n, std::string name) {
	printf("%s: \t", name.c_str());
	for (int i = 0; i < n; i++)
		printf("%.0f, \t", x[i]);
	printf("\n");
}

void generateVector(float *v, int n, int lower, int upper) {
	for (int i = 0; i < n; i++)
		v[i] = (float)(rand() % (upper - lower) + lower);
}

void printMatrix(float *x, int n, int r, int c, std::string name) {
	printf("%s: \n", name.c_str());
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++)
			printf("%.0f, \t", x[i + j * r]);
		printf("\n");
	}
	printf("\n");
}

void printGpuMatrix(float* d_x, int n, int r, int c, std::string name) {
	float* x = new float[n];
    cublasGetVector(n, sizeof(float), d_x, 1, x, 1);
    printMatrix(x, n, r, c, name);
    delete[] x;
}

void printMatrix(int *x, int n, int r, int c, std::string name) {
	printf("%s: \n", name.c_str());
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++)
			printf("%d, \t", x[i + j * r]);
		printf("\n");
	}
	printf("\n");
}

void printGpuMatrix(int* d_x, int n, int r, int c, std::string name) {
	int* x = new int[n];
    cublasGetVector(n, sizeof(int), d_x, 1, x, 1);
    printMatrix(x, n, r, c, name);
    delete[] x;
}
