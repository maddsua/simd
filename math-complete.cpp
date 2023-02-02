#include <iostream>
#include <iomanip>
#include <memory.h>
#include <math.h>


#include <immintrin.h>


/**

     █████  ██   ██  █████       █████  ██    ██  ██████  
    ██   ██  ██ ██  ██   ██     ██   ██ ██    ██ ██       
     █████    ███    █████      ███████ ██    ██ ██   ███ 
    ██   ██  ██ ██  ██   ██     ██   ██  ██  ██  ██    ██ 
     █████  ██   ██  █████      ██   ██   ████    ██████  


	Find average of 8x8 matrix
	
*/

const int32_t matrix_int[8][8] = {
	476, -186, -52, 147, -127, 132, 176, -130,
	314, -18, -164, 361, -237, 285, -124, -212,
	444, 490, -213, 309, 407, 375, 138, 166,
	436, -120, 1, 9, 160, 371, 123, -44,
	-250, 158, 347, 78, 179, -82, 338, -236,
	110, 122, 383, 409, 240, 96, 46, -171,
	154, -3, -20, 246, 377, 125, 296, 227,
	95, 129, 91, 380, 289, 156, -170, -36
};
double matrix_8x8_avg() {

	int32_t result = 0;

	for (size_t m = 0; m < 8; m++) {
		for (size_t n = 0; n < 8; n++) {
			result += matrix_int[m][n];
		}
	}
	
	return ((double)result / 64);
}
double matrix_8x8_avg_sse2() {

	auto vec_rows = _mm_set1_epi32(0);

	for (size_t i = 0; i < 8; i++) {
		auto matrix_A = _mm_loadu_si128((__m128i*)matrix_int[i]);
		auto matrix_A1 = _mm_loadu_si128((__m128i*)(matrix_int[i] + 4));
		auto vect_C = _mm_add_epi32(matrix_A, matrix_A1);

		vec_rows = _mm_add_epi32(vec_rows, vect_C);
	}

	int32_t* rows = (int32_t*)&vec_rows;
	int32_t result = 0;

	for (size_t i = 0; i < 4; i++)
		result += rows[i];
	
	return ((double)result / 64);
}
double matrix_8x8_avg_avx2() {

	auto vec_rows = _mm256_set1_epi32(0);

	for (size_t i = 0; i < 8; i++) {
		auto matrix_A = _mm256_loadu_si256((__m256i*)matrix_int[i]);
		vec_rows = _mm256_add_epi32(vec_rows, matrix_A);
	}

	int32_t* rows = (int32_t*)&vec_rows;
	int32_t result = 0;

	for (size_t i = 0; i < 8; i++)
		result += rows[i];
	
	return ((double)result / 64);
}

/**

    ███████ ██       ██████   █████  ████████      █████  ██   ██  █████  
    ██      ██      ██    ██ ██   ██    ██        ██   ██  ██ ██  ██   ██ 
    █████   ██      ██    ██ ███████    ██         █████    ███    █████  
    ██      ██      ██    ██ ██   ██    ██        ██   ██  ██ ██  ██   ██ 
    ██      ███████  ██████  ██   ██    ██         █████  ██   ██  █████ 

	Multiply two 8x8 matrix of floats, row by row, reverse squre root out of it, then add all vector elements
	
*/
const float matrix_flt[8][8] = {
	2.54, 1.09, 0.32, 0.74, 1.49, 0.53, 1.46, 0.46,
	0.58, 2.44, 0.62, 1.69, 2.30, 1.72, 1.57, 1.68,
	2.33, 1.67, 1.90, 1.84, 0.47, 2.91, 0.26, 2.64,
	1.94, 0.51, 0.17, 2.00, 2.55, 2.71, 2.67, 0.24,
	1.31, 0.19, 0.43, 1.80, 0.10, 2.33, 2.50, 1.60,
	0.64, 2.22, 2.46, 2.34, 0.36, 0.50, 1.64, 1.39,
	1.42, 2.86, 0.96, 1.81, 1.06, 2.57, 0.18, 1.30,
	0.91, 2.25, 1.54, 1.02, 2.84, 0.07, 2.05, 2.43
};

double flt_8x8_mlp() {
	double resrow[8] = {1, 1, 1, 1, 1, 1, 1, 1};
	//	...={1} does not work in GCC 12.2, althouh I remember that it did in TDM-GCC 10.3
	//	not a very big deal but heeey

	for (size_t m = 0; m < 8; m++) {
		for (size_t n = 0; n < 8; n++) {
			resrow[m] *= matrix_flt[n][m];
		}
	}

	for (size_t i = 0; i < 8; i++)
		resrow[i] = 1 / sqrt(resrow[i]);
	
	double result = 1;
	for (size_t i = 0; i < 8; i++)
		result += resrow[i];
	
	return result;
}

double flt_8x8_mlp_sse2() {
	float resrow[8] = {1, 1, 1, 1, 1, 1, 1, 1};

	auto vect_row_A = _mm_set1_ps(1);
	auto vect_row_B = _mm_set1_ps(1);

	for (size_t i = 0; i < 8; i++) {
		vect_row_A = _mm_mul_ps(vect_row_A, _mm_loadu_ps(matrix_flt[i]));
		vect_row_B = _mm_mul_ps(vect_row_B, _mm_loadu_ps(matrix_flt[i] + 4));
	}

	vect_row_A = _mm_rsqrt_ps(vect_row_A);
	vect_row_B = _mm_rsqrt_ps(vect_row_B);

	_mm_storeu_ps(resrow, vect_row_A);
	_mm_storeu_ps(resrow + 4, vect_row_B);

	double result = 1;
	for (size_t i = 0; i < 8; i++)
		result += resrow[i];
	
	return result;
}
double flt_8x8_mlp_avx2() {
	float resrow[8] = {1, 1, 1, 1, 1, 1, 1, 1};

	auto vect_row = _mm256_set1_ps(1);

	for (size_t i = 0; i < 8; i++)
		vect_row = _mm256_mul_ps(vect_row, _mm256_loadu_ps(matrix_flt[i]));

	vect_row = _mm256_rsqrt_ps(vect_row);

	_mm256_storeu_ps(resrow, vect_row);

	double result = 1;
	for (size_t i = 0; i < 8; i++)
		result += resrow[i];
	
	return result;
}


int main() {

	std::cout << matrix_8x8_avg() << "\r\n";
	std::cout << matrix_8x8_avg_sse2() << "\r\n";
	std::cout << matrix_8x8_avg_avx2() << "\r\n\r\n";

	std::cout << flt_8x8_mlp() << "\r\n";
	std::cout << flt_8x8_mlp_sse2() << "\r\n";
	std::cout << flt_8x8_mlp_avx2() << "\r\n";



	return 0;
}