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

int main(void) {
	srand (time(NULL));
	vectorAbsMinMax();
	vectorAbsSum();
	vectorScalar();
	vectorCopyOnDevice();
	return EXIT_SUCCESS;
}
