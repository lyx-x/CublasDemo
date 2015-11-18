#include "level_3.h"

void matrixMultiplication() {
	cusparseHandle_t handle;
	cusparseCreate(&handle);

	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	int m = 3;
	int k = 2;
	int idb = 4;
	int nnz = 4;
	float *val = new float[nnz];
	val[0] = 1; val[1] = 2; val[2] = 3; val[3] = 4;
	int *row = new int[m + 1];
	row[0] = 0; row[1] = 1; row[2] = 2; row[3] = 4;
	int *col = new int[nnz];
	col[0] = 1; col[1] = 0; col[2] = 0; col[3] = 1;
	float *mat = new float[idb * m];
	for (int i = 0; i < idb * m; i++)
		mat[i] = i + 1;
	float *res = new float[k * idb];
	for (int i = 0; i < k * idb; i++)
		res[i] = 1;

	float *d_val;
	cudaMalloc(&d_val, sizeof(float) * nnz);
	cudaMemcpy(d_val, val, size_t(nnz * sizeof(float)), cudaMemcpyHostToDevice);
	int *d_row;
	cudaMalloc(&d_row, sizeof(int) * (m + 1));
	cudaMemcpy(d_row, row, size_t((m + 1) * sizeof(int)), cudaMemcpyHostToDevice);
	int *d_col;
	cudaMalloc(&d_col, sizeof(int) * nnz);
	cudaMemcpy(d_col, col, size_t(nnz * sizeof(int)), cudaMemcpyHostToDevice);
	float *d_mat;
	cudaMalloc(&d_mat, sizeof(float) * idb * m);
	cudaMemcpy(d_mat, mat, size_t(idb * m * sizeof(float)), cudaMemcpyHostToDevice);
	float *d_res;
	cudaMalloc(&d_res, sizeof(float) * k * idb);
	cudaMemcpy(d_res, res, size_t(k * idb * sizeof(float)), cudaMemcpyHostToDevice);

	float one = 1;
	float zero = 0;

	//printGpuMatrix(d_val, nnz, 1, nnz, "val");
	//printGpuMatrix(d_row, m + 1, 1, m + 1, "row");
	//printGpuMatrix(d_col, nnz, 1, nnz, "col");

	float *tmp;
	cudaMalloc(&tmp, sizeof(float) * m * k);
	cusparseScsr2dense(handle, m, k, descr, d_val, d_row, d_col, tmp, m);
	printGpuMatrix(tmp, m * k, m, k, "A");
	cudaFree(tmp);

	printGpuMatrix(d_mat, idb * m, idb, m, "B");

	callCuda(cusparseScsrmm2(handle, CUSPARSE_OPERATION_TRANSPOSE,
			CUSPARSE_OPERATION_TRANSPOSE, m, idb, k, nnz, &one,
			descr, d_val, d_row, d_col, d_mat, idb, &zero, d_res, k));

	printGpuMatrix(d_res, k * idb, k, idb, "C");

	cudaMemcpy(res, d_res, size_t(k * idb * sizeof(float)), cudaMemcpyDeviceToHost);
	//printMatrix(res, k * idb, k, idb, "res");

	cudaFree(d_val);
	cudaFree(d_row);
	cudaFree(d_col);
	cudaFree(d_mat);
	cudaFree(d_res);

	delete[] res;
	delete[] mat;
	delete[] col;
	delete[] row;
	delete[] val;

	cusparseDestroy(handle);
}
