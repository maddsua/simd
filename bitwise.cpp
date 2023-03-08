#include <iostream>
#include <immintrin.h>

#define BLOCK_SIZE		(32)
#define SSE_VECTOR_SZ	(16)


void blockXor(uint8_t* data_A, const uint8_t* data_B) {
	for (size_t i = 0; i < BLOCK_SIZE; i++) {
		data_A[i] ^= data_B[i];
	}
}
void blockXor_sse2(uint8_t* data_A, const uint8_t* data_B) {
	for (size_t i = 0; i < (BLOCK_SIZE / SSE_VECTOR_SZ); i++) {
		auto vec_A = _mm_loadu_si128((__m128i*)(data_A + (i * SSE_VECTOR_SZ)));
		auto vec_B = _mm_loadu_si128((__m128i*)(data_B + (i * SSE_VECTOR_SZ)));
		auto vec_C = _mm_xor_si128(vec_A, vec_B);
		_mm_storeu_si128((__m128i*)(data_A + (i * SSE_VECTOR_SZ)), vec_C);
	}
}
void blockXor_avx2(uint8_t* data_A, const uint8_t* data_B) {
	auto vec_A = _mm256_loadu_si256((__m256i*)data_A);
	auto vec_B = _mm256_loadu_si256((__m256i*)data_B);
	auto vec_C = _mm256_xor_si256(vec_A, vec_B);
	_mm256_storeu_si256((__m256i*)data_A, vec_C);
}


int main() {



	return 0;
}