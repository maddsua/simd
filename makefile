FLAGS	=	-std=c++20 -msse2 -mavx2
# libwinmm is used for measuring time in benchmarks
LIBS	=	-lwinmm


.PHONY: all-before all-after clean clean-custom xor-benchmark xor-usecase hash-usecase arithmetics
all: all-before xor-benchmark all-after

clean:
#	del /S *.o *.exe *.a
	rm -rf *.o *.exe *.a

# make sure that all compiler optimizations are disabled
xor-benchmark:
	g++ xor-benchmark.cpp -o xor-benchmark $(LIBS) $(FLAGS)

xor-usecase:
	g++ xor-usecase.cpp -o xor-usecase $(FLAGS)



hash-function.o: hash-function.cpp
	g++ -c hash-function.cpp -o hash-function.o $(FLAGS)

hash-usecase.o: hash-usecase.cpp 
	g++ -c hash-usecase.cpp -o hash-usecase.o $(FLAGS)

hash-usecase: hash-function.o hash-usecase.o
	g++ hash-usecase.o hash-function.o -o hash-usecase $(FLAGS)

arithmetics:
	g++ arithmetics.cpp -o arithmetics $(LIBS) $(FLAGS)

bitwise:
	g++ bitwise.cpp -o bitwise $(LIBS) $(FLAGS)