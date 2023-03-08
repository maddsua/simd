#ifndef _maddsua_simd_workbench
#define _maddsua_simd_workbench


#include <stdint.h>
#include <stdio.h>


#define TEST_RUNS	(100U)			//	100 test runs
#define TEST_OPS	(100000U)		//	by 100 000 ops


inline void print_hex(const uint8_t str[], size_t size) {
	for(size_t i = 0; i < size; i++)
		printf("%02x", str[i]);
}
inline void print_binary(const uint8_t* data, size_t size) {
	for (size_t m = 0; m < size; m++) {
		for (size_t n = 0; n < 8; n++)
			putc((data[m] & (0b10000000 >> n)) ? '1' : '0', stdout);
		printf(" ");
	}
}

inline time_t avgtime(time_t* array, size_t length) {
	uint64_t temp = 0;
	for (size_t i = 0; i < length; i++) 
		temp += array[i];
	return (temp / length);
}

#endif