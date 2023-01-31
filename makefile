.PHONY: xor-benchmark

# make sure that compiler optimizations are disabled
xor-benchmark:
	g++ xor-benchmark.cpp -o xor-benchmark.exe -mavx2 -lwinmm