/*
 ============================================================================
 Name        : LATest.cpp
 Author      : lyx
 Version     :
 Copyright   : 
 Description : CuBLAS Demo
 ============================================================================
 */

#include <ctime>
#include <cstdlib>
#include "level_1.h"
#include "level_3.h"

int main(void) {
	srand (time(NULL));
	//vectorAbsMinMax();
	//vectorAbsSum();
	//vectorScalar();
	//vectorScalarPlus();
	//vectorCopyOnDevice();
	//vectorDotProduct();
	//vectorNorm();
	//vectorSwap();
	//matrixMultiplication();
	test();
	return EXIT_SUCCESS;
}
