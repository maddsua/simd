#include <iostream>
#include <immintrin.h>
#include <memory.h>
#include <windows.h>
#include <time.h>
#include <array>
#include <fstream>

#include "workbench.hpp"

#define XOR_BLOCK_SIZE	((256)/(8))		//	32

//	totalxor XORes all the bytes inside the buffer to all the other bytes
std::array<uint8_t, XOR_BLOCK_SIZE> totalxor(const uint8_t* data) {

	std::array<uint8_t, XOR_BLOCK_SIZE> result;
	memcpy(result.data(), data, XOR_BLOCK_SIZE);
	for (size_t m = 0; m < XOR_BLOCK_SIZE; m++) {
		for (size_t n = 0; n < XOR_BLOCK_SIZE; n++) {
			result[m] ^= data[n];
		}
	}

	return result;
}
std::array<uint8_t, XOR_BLOCK_SIZE> totalxor_avx2(const uint8_t* data) {

	auto vect_data = _mm256_loadu_si256((const __m256i_u*)data);
	for (size_t i = 0; i < XOR_BLOCK_SIZE; i++) {
		auto vect_xor = _mm256_set1_epi8(data[i]);
		vect_data = _mm256_xor_si256(vect_data, vect_xor);
	}

	std::array<uint8_t, XOR_BLOCK_SIZE> result;
	_mm256_storeu_si256((__m256i_u*)result.data(), vect_data);

	return result;
}
std::array<uint8_t, XOR_BLOCK_SIZE> totalxor_sse(const uint8_t* data) {

	__m128i_u vect_data[2] = {
		_mm_loadu_si128((const __m128i_u*)data),
		_mm_loadu_si128((const __m128i_u*)(data + (XOR_BLOCK_SIZE / 2)))
	};

	for (size_t m = 0; m < XOR_BLOCK_SIZE; m++) {
		auto vect_xor = _mm_set1_epi8(data[m]);
		vect_data[0] = _mm_xor_si128(vect_data[0], vect_xor);
		vect_data[1] = _mm_xor_si128(vect_data[1], vect_xor);
	}

	std::array<uint8_t, XOR_BLOCK_SIZE> result;
	_mm_storeu_si128((__m128i_u*)result.data(), vect_data[0]);
	_mm_storeu_si128((__m128i_u*)(result.data() + (XOR_BLOCK_SIZE / 2)), vect_data[1]);

	return result;
}

int main() {

	std::cout << "This test is gonna perform " << TEST_RUNS << " runs of " << TEST_OPS << " XOR operations on 256-bit buffers\r\n";
	std::cout << "Test has been started...\r\n\r\n";
	
	const std::array<uint8_t, XOR_BLOCK_SIZE> dataBlock = {156,252,14,198,96,30,193,195,143,159,237,175,168,57,210,42,10,6,55,236,246,92,66,62,139,123,5,203,47,172,194,93};

	time_t timer;
	std::array<time_t, TEST_RUNS> test_ctrl;
	std::array<time_t, TEST_RUNS> test_sse;
	std::array<time_t, TEST_RUNS> test_avx2;

	//	test without any simd
	for (size_t m = 0; m < TEST_RUNS; m++) {
		timer = timeGetTime();
		for (size_t n = 0; n < TEST_OPS; n++) {
			auto result = totalxor(dataBlock.data());
		}
		test_ctrl[m] = timeGetTime() - timer;
	}

	std::cout << "Control test ended. AVG: " << avgtime(test_ctrl.data(), TEST_RUNS) << "ms/block\r\n";


	//	test with sse
	for (size_t m = 0; m < TEST_RUNS; m++) {
		timer = timeGetTime();
		for (size_t n = 0; n < TEST_OPS; n++) {
			auto result = totalxor_sse(dataBlock.data());
		}
		test_sse[m] = timeGetTime() - timer;
	}

	std::cout << "SSE test ended. AVG: " << avgtime(test_sse.data(), TEST_RUNS) << "ms/block\r\n";


	//	test with avx2
	for (size_t m = 0; m < TEST_RUNS; m++) {
		timer = timeGetTime();
		for (size_t n = 0; n < TEST_OPS; n++) {
			auto result = totalxor_avx2(dataBlock.data());
		}
		test_avx2[m] = timeGetTime() - timer;
	}

	std::cout << "AVX2 test ended. AVG: " << avgtime(test_avx2.data(), TEST_RUNS) << "ms/block\r\n";

	//	save test data
	std::cout << "\r\nWriting test data to .csv...\r\n";
	
	std::string filename = std::string("benchmarks-data/") + "benchmark_xor_" + std::to_string(time(nullptr)) + ".csv";
	std::ofstream output(filename, std::ios::out);

	output << "Control,SSE,AVX2,Unit" << "\n";
	for (size_t i = 0; i < TEST_RUNS; i++){
		output << test_ctrl[i] << "," << test_sse[i] << "," << test_avx2[i] << ",ms/" << TEST_OPS << " ops\n";
	}

	output.close();

	return 0;
}