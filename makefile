.PHONY: xor-benchmark xor-usecase

# make sure that compiler optimizations are disabled
xor-benchmark:
	g++ xor-benchmark.cpp -o xor-benchmark.exe -mavx2 -lwinmm

xor-usecase:
	g++ xor-usecase.cpp -o xor-usecase.exe