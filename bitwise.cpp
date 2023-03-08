#include <iostream>
#include <array>
#include <immintrin.h>
#include <windows.h>
#include <fstream>

#include "workbench.hpp"

#define BLOCK_SIZE		(32)
#define DEMO_BLOCK_SIZE	(8)
#define SSE_VECTOR_SZ	(16)
#define AVX2_VECTOR_SZ	(32)


std::array<uint8_t, BLOCK_SIZE> blockXor(const uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	for (size_t i = 0; i < BLOCK_SIZE; i++) {
		result[i] = data_A[i] ^ data_B[i];
	}
	return result;
}
std::array<uint8_t, BLOCK_SIZE> blockXor_sse2(uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	for (size_t i = 0; i < (BLOCK_SIZE / SSE_VECTOR_SZ); i++) {
		auto vec_A = _mm_loadu_si128((__m128i*)(data_A + (i * SSE_VECTOR_SZ)));
		auto vec_B = _mm_loadu_si128((__m128i*)(data_B + (i * SSE_VECTOR_SZ)));
		auto vec_C = _mm_xor_si128(vec_A, vec_B);
		_mm_storeu_si128((__m128i*)(result.data() + (i * SSE_VECTOR_SZ)), vec_C);
	}
	return result;
}
std::array<uint8_t, BLOCK_SIZE> blockXor_avx2(uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	auto vec_A = _mm256_loadu_si256((__m256i*)data_A);
	auto vec_B = _mm256_loadu_si256((__m256i*)data_B);
	auto vec_C = _mm256_xor_si256(vec_A, vec_B);
	_mm256_storeu_si256((__m256i*)result.data(), vec_C);
	return result;
}
std::array<uint8_t, DEMO_BLOCK_SIZE> blockXor_demo(const uint8_t* data, const uint8_t* key) {

	std::array<uint8_t, DEMO_BLOCK_SIZE> result;
	for (size_t i = 0; i < DEMO_BLOCK_SIZE; i++)
		result[i] = data[i] ^ key[i];

	return result;
}


std::array<uint8_t, BLOCK_SIZE> blockAnd(const uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	for (size_t i = 0; i < BLOCK_SIZE; i++) {
		result[i] = data_A[i] & data_B[i];
	}
	return result;
}
std::array<uint8_t, BLOCK_SIZE> blockAnd_sse2(uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	for (size_t i = 0; i < (BLOCK_SIZE / SSE_VECTOR_SZ); i++) {
		auto vec_A = _mm_loadu_si128((__m128i*)(data_A + (i * SSE_VECTOR_SZ)));
		auto vec_B = _mm_loadu_si128((__m128i*)(data_B + (i * SSE_VECTOR_SZ)));
		auto vec_C = _mm_and_si128(vec_A, vec_B);
		_mm_storeu_si128((__m128i*)(result.data() + (i * SSE_VECTOR_SZ)), vec_C);
	}
	return result;
}
std::array<uint8_t, BLOCK_SIZE> blockAnd_avx2(uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	auto vec_A = _mm256_loadu_si256((__m256i*)data_A);
	auto vec_B = _mm256_loadu_si256((__m256i*)data_B);
	auto vec_C = _mm256_and_si256(vec_A, vec_B);
	_mm256_storeu_si256((__m256i*)result.data(), vec_C);
	return result;
}


std::array<uint8_t, BLOCK_SIZE> blockOr(const uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	for (size_t i = 0; i < BLOCK_SIZE; i++) {
		result[i] = data_A[i] | data_B[i];
	}
	return result;
}
std::array<uint8_t, BLOCK_SIZE> blockOr_sse2(uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	for (size_t i = 0; i < (BLOCK_SIZE / SSE_VECTOR_SZ); i++) {
		auto vec_A = _mm_loadu_si128((__m128i*)(data_A + (i * SSE_VECTOR_SZ)));
		auto vec_B = _mm_loadu_si128((__m128i*)(data_B + (i * SSE_VECTOR_SZ)));
		auto vec_C = _mm_or_si128(vec_A, vec_B);
		_mm_storeu_si128((__m128i*)(result.data() + (i * SSE_VECTOR_SZ)), vec_C);
	}
	return result;
}
std::array<uint8_t, BLOCK_SIZE> blockOr_avx2(uint8_t* data_A, const uint8_t* data_B) {
	std::array<uint8_t, BLOCK_SIZE> result;
	auto vec_A = _mm256_loadu_si256((__m256i*)data_A);
	auto vec_B = _mm256_loadu_si256((__m256i*)data_B);
	auto vec_C = _mm256_or_si256(vec_A, vec_B);
	_mm256_storeu_si256((__m256i*)result.data(), vec_C);
	return result;
}


void invalidResult() {
	std::cout << "Error while performing operation\r\n";
	exit(1);
}


int main() {

	/*
		Usecase
	*/
	std::cout << "\r\nBitwise operations are used literally everywhere, especially heavily in cryptography/encryption.\r\n";
	std::cout << "\r\nVery basic encryption example using XOR on a block of data:\r\n";

	std::array<uint8_t, DEMO_BLOCK_SIZE> data = {'t', 'e', 's', 't', 'd', 'a', 't', 'a'};
	std::array<uint8_t, DEMO_BLOCK_SIZE> key = {'k', 'e', 'y', '_', 'h', 'e', 'r', 'e'};

	//	print data
	std::cout << "Data:\r\n";
	print_hex(data.data(), DEMO_BLOCK_SIZE);
	std::cout << "\r\n";

	//	print key
	std::cout << "Key:\r\n";
	print_hex(key.data(), DEMO_BLOCK_SIZE);
	std::cout << "\r\n";

	//	XOR and print
	auto xored = blockXor_demo(data.data(), key.data());

	std::cout << "XORed data:\r\n";
	print_hex(xored.data(), DEMO_BLOCK_SIZE);
	std::cout << "\r\n";

	//	XOR data with the key back to restore it
	auto restored = blockXor_demo(xored.data(), key.data());

	std::cout << "Restored data:\r\n";
	print_hex(restored.data(), DEMO_BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	report results
	std::cout << ((restored == data) ? "Data matched" : "ERROR: Data didn't match") << "\r\n\r\n";


	/*
		Benchmark
	*/
	uint8_t bin_data[2][BLOCK_SIZE] = {
		160,173,60,01,198,57,18,66,80,131,241,20,87,146,97,78,195,110,8,58,220,61,103,4,84,162,229,215,111,246,86,88,
		224,212,254,15,45,112,184,0,9,48,133,81,42,253,31,116,167,118,171,145,243,207,83,214,21,237,175,44,155,199,142,11
	};

	std::cout << "Press any key to start the benchmark...\r\n";
	system("pause");
	std::cout << "Running...\r\n";

	time_t timer;
	
	//	XOR benchmark
	std::array<time_t, TEST_RUNS> test1_ctrl;
	std::array<time_t, TEST_RUNS> test1_sse2;
	std::array<time_t, TEST_RUNS> test1_avx2;
	{
		auto refResult = blockXor(bin_data[0], bin_data[1]);

		//	test without any simd
		std::cout << "XOR test control run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockXor(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test1_ctrl[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test1_ctrl.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";


		//	test with sse
		std::cout << "XOR teset SSE run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockXor_sse2(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test1_sse2[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test1_sse2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";

		//	test with avx2
		std::cout << "XOR teset AVX2 run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockXor_avx2(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test1_avx2[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test1_avx2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";
	}

	//	AND benchmark
	std::array<time_t, TEST_RUNS> test2_ctrl;
	std::array<time_t, TEST_RUNS> test2_sse2;
	std::array<time_t, TEST_RUNS> test2_avx2;
	{
		auto refResult = blockAnd(bin_data[0], bin_data[1]);

		//	test without any simd
		std::cout << "AND test control run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockAnd(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test2_ctrl[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test2_ctrl.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";


		//	test with sse
		std::cout << "AND teset SSE run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockAnd_sse2(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test2_sse2[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test2_sse2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";

		//	test with avx2
		std::cout << "AND teset AVX2 run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockAnd_avx2(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test2_avx2[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test2_avx2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";
	}

	//	OR benchmark
	std::array<time_t, TEST_RUNS> test3_ctrl;
	std::array<time_t, TEST_RUNS> test3_sse2;
	std::array<time_t, TEST_RUNS> test3_avx2;
	{
		auto refResult = blockOr(bin_data[0], bin_data[1]);

		//	test without any simd
		std::cout << "OR test control run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockOr(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test3_ctrl[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test3_ctrl.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";


		//	test with sse
		std::cout << "OR teset SSE run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockOr_sse2(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test3_sse2[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test3_sse2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";

		//	test with avx2
		std::cout << "OR teset AVX2 run...";
		for (size_t m = 0; m < TEST_RUNS; m++) {
			timer = timeGetTime();
			for (size_t n = 0; n < TEST_OPS_RED; n++) {
				auto opResult = blockOr_avx2(bin_data[0], bin_data[1]);
				if (opResult != refResult) invalidResult();
			}
			test3_avx2[m] = timeGetTime() - timer;
		}
		std::cout << " AVG: " << avgtime(test3_avx2.data(), TEST_RUNS) << "ms/" << TEST_OPS_RED <<"ops\r\n";
	}

	//	save test data
	std::string filename = std::string("benchmarks-data/") + "benchmark_bitwise_" + std::to_string(time(nullptr)) + ".csv";
	std::cout << "\r\nTest ended. Writing data to " << filename << std::endl;
	std::ofstream output(filename, std::ios::out);

	output << "XOR-Control,XOR-SSE,XOR-AVX2,"
			<< "AND-Control,AND-SSE,AND-AVX2,"
			<< "OR-Control,OR-SSE,OR-AVX2,"
			<< "Unit (ms/n ops)\n";

	for (size_t i = 0; i < TEST_RUNS; i++){
		output << test1_ctrl[i] << "," << test1_sse2[i] << "," << test1_avx2[i] << ","
				<< test2_ctrl[i] << "," << test2_sse2[i] << "," << test2_avx2[i] << ","
				<< test3_ctrl[i] << "," << test3_sse2[i] << "," << test3_avx2[i] << ","
				<< TEST_OPS << "\n";
	}

	output.close();



	/*auto x = blockOr(bin_data[0], bin_data[1]);
	print_hex(x.data(), BLOCK_SIZE);
	std::cout << std::endl;


	x = blockOr_sse2(bin_data[0], bin_data[1]);
	print_hex(x.data(), BLOCK_SIZE);
	std::cout << std::endl;


	x = blockOr_avx2(bin_data[0], bin_data[1]);
	print_hex(x.data(), BLOCK_SIZE);
	std::cout << std::endl;*/



	return 0;
}