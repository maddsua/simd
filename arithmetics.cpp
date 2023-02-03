#include <iostream>
#include <iomanip>
#include <memory.h>
#include <math.h>
#include <array>
#include <windows.h>
#include <fstream>

#include <immintrin.h>

#include "workbench.hpp"


void print_matrixF(float* matrix, size_t dim) {

	const size_t fullSize = (dim * dim);

	for (size_t m = 0; m < fullSize; m += dim) {
		for (size_t n = 0; n < dim; n++) {
			std:: cout << std::setw(10) << std::left << matrix[m + n] << " ";
		}
		std::cout << std::endl;
	}
}
void print_matrixD(double* matrix, size_t dim) {

	const size_t fullSize = (dim * dim);

	for (size_t m = 0; m < fullSize; m += dim) {
		for (size_t n = 0; n < dim; n++) {
			std:: cout << std::setw(10) << std::left << matrix[m + n] << " ";
		}
		std::cout << std::endl;
	}
}



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

	int32_t rows[4];
	_mm_storeu_si128((__m128i*)rows, vec_rows);

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

	int32_t rows[8];
	_mm256_storeu_si256((__m256i*)rows, vec_rows);

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

/**

    ██████   ██████  ██    ██ ██████  ██      ███████     ██████  ███████ ███    ██
    ██   ██ ██    ██ ██    ██ ██   ██ ██      ██          ██   ██ ██      ████   ██
    ██   ██ ██    ██ ██    ██ ██████  ██      █████       ██████  █████   ██ ██  ██
    ██   ██ ██    ██ ██    ██ ██   ██ ██      ██          ██      ██      ██  ██ ██
    ██████   ██████   ██████  ██████  ███████ ███████     ██      ███████ ██   ████ ██ ██ ██

	It's time to actually fck the crp out of your PC!

	Ok I'm joking, this only run on single core, so the PC would be totally fine
	
*/

const double matrix_dbl[8][8] = {
	0.02494159, 0.71868752, 0.91990175, 0.34410689, 0.47312923, 0.30208912, 0.49589867, 0.02516413,
	0.88883377, 0.06795215, 0.08575813, 0.22577430, 0.49062035, 0.86974825, 0.32800815, 0.12455937,
	0.47718442, 0.57817785, 0.65302181, 0.64077491, 0.00368169, 0.84689647, 0.37906349, 0.25878278,
	0.32721652, 0.64109163, 0.97700378, 0.98086717, 0.21519017, 0.14436138, 0.76528899, 0.61312114,
	0.26716477, 0.76112935, 0.00952823, 0.51584315, 0.92180513, 0.37365493, 0.28754857, 0.62823112,
	0.39187170, 0.37756994, 0.14195147, 0.83827942, 0.93483692, 0.54641417, 0.11881003, 0.76125618,
	0.35885252, 0.30131080, 0.10792290, 0.87547434, 0.40315502, 0.56225633, 0.50602718, 0.99637334,
	0.85471644, 0.74252133, 0.79062605, 0.26826468, 0.47558618, 0.92064569, 0.50718114, 0.82362572
};

std::array <float, 16> d8_f4() {

	double tmpmtrx[8][8];
	for (size_t m = 0; m < 8; m++) {
		for (size_t n = 0; n < 8; n++) {
			tmpmtrx[m][n] = 1 / sqrt(matrix_dbl[m][n]);
		}
	}

	double tmpmtrx2[8][8];
	memset(tmpmtrx2, 0, 64 * sizeof(double));
	for (size_t m = 0; m < 4; m++) {
		for (size_t n = 0; n < 8; n++) {
			tmpmtrx2[m][n] = tmpmtrx[(m * 2)][n] * tmpmtrx[(m * 2) + 1][n];
		}
	}

	std::array <float, 16> result;

	float tmpmtrx3[4][4];
	memset(tmpmtrx3, 0, 16 * sizeof(float));
	for (size_t m = 0; m < 4; m++) {
		for (size_t n = 0; n < 4; n++) {
			tmpmtrx3[m][n] = 1 / (sqrt(pow(tmpmtrx2[m][(n * 2)], 2) / pow(tmpmtrx2[m][(n * 2) + 1], 2)));
		}
	}

	memcpy(result.data(), tmpmtrx3, 16 * sizeof(float));

	return result;
}
std::array <float, 16> d8_f4_sse2() {

	double tmpmtrx[8][8];
	for (size_t m = 0; m < 8; m++) {
		for (size_t n = 0; n < 4; n++) {
			auto vect_rsqrt = _mm_sqrt_pd(_mm_loadu_pd(&matrix_dbl[m][n * 2]));
			auto divided = _mm_div_pd(_mm_set_pd1(1), vect_rsqrt);
			_mm_storeu_pd(&tmpmtrx[m][n * 2], divided);
		}
	}

	double tmpmtrx2[8][8];	
	double tmpmtrx2_A[8];
	double tmpmtrx2_B[8];
	
	for (size_t m = 0; m < 4; m++) {

		const size_t _2m = m * 2;

		//	pack inputs to two arrays
		for (size_t n = 0; n < 8; n++) {
			tmpmtrx2_A[n] = tmpmtrx[_2m][n];
			tmpmtrx2_B[n] = tmpmtrx[_2m + 1][n];
		}

		//	multiply the input doubles
		for (size_t n = 0; n < 4; n++) {
			const size_t _2n = n * 2;
			auto vect_A = _mm_loadu_pd(&tmpmtrx2_A[_2n]);
			auto vect_B = _mm_loadu_pd(&tmpmtrx2_B[_2n]);
			auto multiplied = _mm_mul_pd(vect_A, vect_B);
			_mm_storeu_pd(&tmpmtrx2[m][_2n], multiplied);
		}
	}

	std::array <float, 16> result;

	float tmpmtrx3[4][4];
	double tmpmtrx3_A[4];
	double tmpmtrx3_B[4];

	for (size_t m = 0; m < 4; m++) {

		//	pack inputs again
		for (size_t n = 0; n < 4; n++) {
			const size_t _2n = 2 * n;
			tmpmtrx3_A[n] = tmpmtrx2[m][_2n];
			tmpmtrx3_B[n] = tmpmtrx2[m][_2n + 1];
		}

		//	perform math on vectors		
		for (size_t n = 0; n < 2; n++) {
			const size_t _2n = n * 2;
			
			auto vect_A = _mm_loadu_pd(&tmpmtrx3_A[_2n]);
			auto vect_B = _mm_loadu_pd(&tmpmtrx3_B[_2n]);

			auto vect_pow_A = _mm_mul_pd(vect_A, vect_A);
			auto vect_pow_B = _mm_mul_pd(vect_B, vect_B);

			auto vect_divd = _mm_div_pd(vect_pow_A, vect_pow_B);
			auto vect_sqrt = _mm_sqrt_pd(vect_divd);

			auto vect_invsqrt = _mm_div_pd(_mm_set_pd1(1), vect_sqrt);

			_mm_storeu_ps(&tmpmtrx3[m][_2n], _mm_cvtpd_ps(vect_invsqrt));
		}
	}

	memcpy(result.data(), tmpmtrx3, 16 * sizeof(float));

	return result;
}
std::array <float, 16> d8_f4_avx2() {

	double tmpmtrx[8][8];
	for (size_t m = 0; m < 8; m++) {
		for (size_t n = 0; n < 2; n++) {
			auto vect_rsqrt = _mm256_sqrt_pd(_mm256_loadu_pd(&matrix_dbl[m][n * 4]));
			auto divided = _mm256_div_pd(_mm256_set_pd(1, 1, 1, 1), vect_rsqrt);
			_mm256_storeu_pd(&tmpmtrx[m][n * 4], divided);
		}
	}

	double tmpmtrx2[8][8];	
	double tmpmtrx2_A[8];
	double tmpmtrx2_B[8];
	
	for (size_t m = 0; m < 4; m++) {
		const size_t _2m = m * 2;
		//	pack inputs to two arrays
		for (size_t n = 0; n < 8; n++) {
			tmpmtrx2_A[n] = tmpmtrx[_2m][n];
			tmpmtrx2_B[n] = tmpmtrx[_2m + 1][n];
		}
		//	multiply the input doubles
		for (size_t n = 0; n < 2; n++) {
			const size_t _4n = n * 4;
			auto vect_A = _mm256_loadu_pd(&tmpmtrx2_A[_4n]);
			auto vect_B = _mm256_loadu_pd(&tmpmtrx2_B[_4n]);
			auto multiplied = _mm256_mul_pd(vect_A, vect_B);
			_mm256_storeu_pd(&tmpmtrx2[m][_4n], multiplied);
		}
	}

	std::array <float, 16> result;

	float tmpmtrx3[4][4];
	double tmpmtrx3_A[4];
	double tmpmtrx3_B[4];

	for (size_t m = 0; m < 4; m++) {
		
		//	pack inputs again
		for (size_t n = 0; n < 4; n++) {
			const size_t _2n = 2 * n;
			tmpmtrx3_A[n] = tmpmtrx2[m][_2n];
			tmpmtrx3_B[n] = tmpmtrx2[m][_2n + 1];
		}

		//	perform math on vectors
		auto vect_A = _mm256_loadu_pd(tmpmtrx3_A);
		auto vect_B = _mm256_loadu_pd(tmpmtrx3_B);

		auto vect_pow_A = _mm256_mul_pd(vect_A, vect_A);
		auto vect_pow_B = _mm256_mul_pd(vect_B, vect_B);

		auto vect_divd = _mm256_div_pd(vect_pow_A, vect_pow_B);
		auto vect_sqrt = _mm256_sqrt_pd(vect_divd);

		auto vect_invsqrt = _mm256_div_pd(_mm256_set_pd(1, 1, 1, 1), vect_sqrt);

		_mm_storeu_ps(tmpmtrx3[m], _mm256_cvtpd_ps(vect_invsqrt));
	}

	memcpy(result.data(), tmpmtrx3, 16 * sizeof(float));

	return result;
}

/**

    ██████  ██  ██████      ██ ███    ██ ████████
    ██   ██ ██ ██           ██ ████   ██    ██
    ██████  ██ ██   ███     ██ ██ ██  ██    ██
    ██   ██ ██ ██    ██     ██ ██  ██ ██    ██
    ██████  ██  ██████      ██ ██   ████    ██

	64-bit integers ops
	
*/

const int64_t matrix_bigint[8][4] {
	-779, -45579, 70910, -26100,
	72448, 16274, 48480, -6477,
	11284, -32778, 32936, -92112,
	13822, -89155, 41849, 84882,
	55442, 15198, 77350, -80102,
	61894, 25507, 97139, -1519,
	-53315, 17583, -39452, 49882,
	57465, 56360, -52681, 28063
};

int64_t bigint_calc() {

	int64_t temp[4];

	//	multiply first row
	for (size_t i = 0; i < 4; i++) temp[i] = matrix_bigint[0][i] * matrix_bigint[1][i];

	//	add
	for (size_t i = 0; i < 4; i++) temp[i] += matrix_bigint[2][i];

	//	substract
	for (size_t i = 0; i < 4; i++) temp[i] -= matrix_bigint[3][i];

	//	divide
	//for (size_t i = 0; i < 4; i++) temp[i] /= matrix_bigint[4][i];

	//	make absolute
	//for (size_t i = 0; i < 4; i++) temp[i] = _abs64(temp[i]);

	//	find inverse-of-next-row square root
	//for (size_t i = 0; i < 4; i++) temp[i] = matrix_bigint[5][i] / sqrtl(temp[i]);

	//	sum and return
	int64_t result = 0;
	for (size_t i = 0; i < 4; i++) result += temp[i];
	
	return result;
}
int64_t bigint_calc_sse2() {

	int64_t temp[4];

	//	multiply first row
	//	just no 64intx64int op in SSE
	for (size_t i = 0; i < 4; i++) temp[i] = matrix_bigint[0][i] * matrix_bigint[1][i];

	//	add
	auto vec_A = _mm_add_epi64(_mm_loadu_si128((__m128i*)temp), _mm_loadu_si128((__m128i*)matrix_bigint[2]));
	auto vec_B = _mm_add_epi64(_mm_loadu_si128((__m128i*)(temp + 2)), _mm_loadu_si128((__m128i*)(matrix_bigint[2] + 2)));

	//	substract
	vec_A = _mm_sub_epi64(vec_A, _mm_loadu_si128((__m128i*)matrix_bigint[3]));
	vec_B = _mm_sub_epi64(vec_B, _mm_loadu_si128((__m128i*)(matrix_bigint[3] + 2)));

	_mm_storeu_si128((__m128i*)temp, vec_A);
	_mm_storeu_si128((__m128i*)(temp + 2), vec_B);

	//	divide
	//	no division op, again
	//for (size_t i = 0; i < 4; i++) temp[i] /= matrix_bigint[4][i];

	//	make absolute
	//	ahhh no abs too
	//for (size_t i = 0; i < 4; i++) temp[i] = _abs64(temp[i]);

	//	find inverse-of-next-row square root
	//	oops, no sqrt too =/
	//for (size_t i = 0; i < 4; i++) temp[i] = matrix_bigint[5][i] / sqrtl(temp[i]);

	//	sum and return
	int64_t result = 0;
	for (size_t i = 0; i < 4; i++) result += temp[i];
	
	return result;
}

int64_t bigint_calc_avx2() {

	int64_t temp[4];

	//	multiply first row
	//	yeah yeah to 64 int multiply instruction even in AVX2
	for (size_t i = 0; i < 4; i++) temp[i] = matrix_bigint[0][i] * matrix_bigint[1][i];

	//	add
	auto vector = _mm256_add_epi64(_mm256_loadu_si256((__m256i_u*)temp), _mm256_loadu_si256((__m256i_u*)matrix_bigint[2]));

	//	substract
	vector = _mm256_sub_epi64(vector, _mm256_loadu_si256((__m256i_u*)matrix_bigint[3]));

	_mm256_storeu_si256((__m256i_u*)temp, vector);

	//	divide
	//	no division op, oooopsie again x2
	//for (size_t i = 0; i < 4; i++) temp[i] /= matrix_bigint[4][i];

	//	make absolute
	//	only in avx512, kids
	//for (size_t i = 0; i < 4; i++) temp[i] = _abs64(temp[i]);

	//	find inverse-of-next-row square root
	//	aaaaaaaaah, no square root for 64-bit ints. ok Inter, got it
	//for (size_t i = 0; i < 4; i++) temp[i] = matrix_bigint[5][i] / sqrtl(temp[i]);

	//	sum and return
	int64_t result = 0;
	for (size_t i = 0; i < 4; i++) result += temp[i];
	
	return result;
}

void invalidResult() {
	std::cout << "Error while performing operation\r\n";
	exit(1);
}

int main() {

	std::cout << "\r\n\r\nThis demo is gonna performs some arithmetic computations along with benchmarks\r\n";
	std::cout << "Running " << TEST_OPS << " instances of each operation...\r\n";

	auto timer = timeGetTime();

	//goto test4;

	test1:
	//	average of 8x8 32-bit int matrix
	std::array<time_t, TEST_RUNS> test1_ctrl;
	std::array<time_t, TEST_RUNS> test1_sse2;
	std::array<time_t, TEST_RUNS> test1_avx2;
	{
		std::cout << "\r\n\r\nTest 1. Average value of 8x8 32-bit int matrix...\r\n";

		//	no simd
		auto refResult = matrix_8x8_avg();
		std::cout << "Expected computation result: " << refResult << "\r\n\r\n";
		std::cout << "Control run... ";
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = matrix_8x8_avg();
				if (opResult != refResult) invalidResult();
			}
			test1_ctrl[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test1_ctrl.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";

		//	sse2
		std::cout << "SSE2 run... ";
		refResult = matrix_8x8_avg_sse2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = matrix_8x8_avg_sse2();
				if (opResult != refResult) invalidResult();
			}
			test1_sse2[r] = timeGetTime() - timer;
		}
		std::cout << "AVG " << avgtime(test1_sse2.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";

		//	avx2
		std::cout << "AVX run... ";
		refResult = matrix_8x8_avg_avx2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = matrix_8x8_avg_avx2();
				if (opResult != refResult) invalidResult();
			}
			test1_avx2[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test1_avx2.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";
	}

	test2:
	//	double 8x8 to float 4x4 matrix transform
	std::array<time_t, TEST_RUNS> test2_ctrl;
	std::array<time_t, TEST_RUNS> test2_sse2;
	std::array<time_t, TEST_RUNS> test2_avx2;
	{
		std::cout << "\r\n\r\nTest 2. Float 8x8 to single float matrix multiplication...\r\n";

		//	no simd
		auto refResult = flt_8x8_mlp();
		std::cout << "Expected computation result: " << refResult << "\r\n\r\n";
		std::cout << "Control run... ";
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = flt_8x8_mlp();
				if (opResult != refResult) invalidResult();
			}
			test2_ctrl[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test2_ctrl.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";

		//	sse2
		std::cout << "SSE2 run... ";
		refResult = flt_8x8_mlp_sse2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = flt_8x8_mlp_sse2();
				if (opResult != refResult) invalidResult();
			}
			test2_sse2[r] = timeGetTime() - timer;
		}
		std::cout << "AVG " << avgtime(test2_sse2.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";

		//	avx2
		std::cout << "AVX run... ";
		refResult = flt_8x8_mlp_avx2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = flt_8x8_mlp_avx2();
				if (opResult != refResult) invalidResult();
			}
			test2_avx2[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test2_avx2.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";
	}

	test3:
	//	double 8x8 to float 4x4 matrix transform
	//	this is a computationaly intensive test, number of samples is lowered by a factor of 10
	std::array<time_t, TEST_RUNS> test3_ctrl;
	std::array<time_t, TEST_RUNS> test3_sse2;
	std::array<time_t, TEST_RUNS> test3_avx2;
	{
		std::cout << "\r\n\r\nTest 3. Double 8x8 to float 4x4 matrix transform...\r\n";

		//	no simd
		auto refResult = d8_f4();

		std::cout << "Expected computation result:\r\n\r\n";
		print_matrixF(refResult.data(), 4);
		std::cout << "\r\n\r\n";

		std::cout << "Control run... ";
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS_RED; i++) {
				auto opResult = d8_f4();
				if (opResult != refResult) invalidResult();
			}
			test3_ctrl[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test3_ctrl.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED << "ops\r\n";

		//	sse2
		std::cout << "SSE2 run... ";
		refResult = d8_f4_sse2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS_RED; i++) {
				auto opResult = d8_f4_sse2();
				if (opResult != refResult) invalidResult();
			}
			test3_sse2[r] = timeGetTime() - timer;
		}
		std::cout << "AVG " << avgtime(test3_sse2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED << "ops\r\n";

		//	avx2
		std::cout << "AVX run... ";
		refResult = d8_f4_avx2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS_RED; i++) {
				auto opResult = d8_f4_avx2();
				if (opResult != refResult) invalidResult();
			}
			test3_avx2[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test3_avx2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED << "ops\r\n";
	}

	test4:
	//	bitg int test
	//	irrelevant one due to weak vectorization
	std::array<time_t, TEST_RUNS> test4_ctrl;
	std::array<time_t, TEST_RUNS> test4_sse2;
	std::array<time_t, TEST_RUNS> test4_avx2;
	{
		std::cout << "\r\n\r\nTest 4. Just checking how fast you can crunch the 64-bit integers...\r\n";

		//	no simd
		auto refResult = bigint_calc();
		std::cout << "Expected computation result: " << refResult << "\r\n\r\n";
		std::cout << "Control run... ";
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = bigint_calc();
				if (opResult != refResult) invalidResult();
			}
			test4_ctrl[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test4_ctrl.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";

		//	sse2
		std::cout << "SSE2 run... ";
		refResult = bigint_calc_sse2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = bigint_calc_sse2();
				if (opResult != refResult) invalidResult();
			}
			test4_sse2[r] = timeGetTime() - timer;
		}
		std::cout << "AVG " << avgtime(test4_sse2.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";

		//	avx2
		std::cout << "AVX run... ";
		refResult = bigint_calc_avx2();
		for (size_t r = 0; r < TEST_RUNS; r++) {
			timer = timeGetTime();
			for (size_t i = 0; i < TEST_OPS; i++) {
				auto opResult = bigint_calc_avx2();
				if (opResult != refResult) invalidResult();
			}
			test4_avx2[r] = timeGetTime() - timer;
		}

		std::cout << "AVG " << avgtime(test4_avx2.data(), TEST_RUNS) << "ms/" << TEST_OPS << "ops\r\n";
	}

	//	save test data
	std::string filename = std::string("benchmarks-data/") + "benchmark_arithmetics_" + std::to_string(time(nullptr)) + ".csv";
	std::cout << "\r\nTest ended. Writing data to " << filename << std::endl;
	std::ofstream output(filename, std::ios::out);

	output << "int32-Control,int32-SSE,int32-AVX2,"
			<<  "float-Control,float-SSE,float-AVX2,"
			<< "double-Control,double-SSE,double-AVX2,"
			<< "int64-Control,int64-SSE,int64-AVX2,"
			<< "Unit (ms/n ops)\n";

	for (size_t i = 0; i < TEST_RUNS; i++){
		output << test1_ctrl[i] << "," << test1_sse2[i] << "," << test1_avx2[i] << ","
				<< test2_ctrl[i] << "," << test2_sse2[i] << "," << test2_avx2[i] << ","
				<< test3_ctrl[i] << "," << test3_sse2[i] << "," << test3_avx2[i] << ","
				<< test4_ctrl[i] << "," << test4_sse2[i] << "," << test4_avx2[i]  << ","
				<< TEST_OPS << "\n";
	}

	output.close();

	return 0;
}