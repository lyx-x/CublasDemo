#include "helper.h"

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
