#ifndef __cpu_capabilities
#define __cpu_capabilities

	struct cpuintrinsics {
		//	Misc.
		bool MMX;
		bool X64;
		bool ABM;
		bool RDRAND;
		bool BMI1;
		bool BMI2;
		bool ADX;
		bool PREFETCHWT1;

		//	SIMD: 128-bit
		bool SSE;
		bool SSE2;
		bool SSE3;
		bool SSSE3;
		bool SSE41;
		bool SSE42;
		bool SSE4A;
		bool AES;
		bool SHA;

		//	SIMD: 256-bit
		bool AVX;
		bool XOP;
		bool FMA3;
		bool FMA4;
		bool AVX2;

		//	SIMD: 512-bit
		bool AVX512F;		//	AVX512 Foundation
		bool AVX512CD;		//	AVX512 Conflict Detection
		bool AVX512PF;		//	AVX512 Prefetch
		bool AVX512ER; 		//	AVX512 Exponential + Reciprocal
		bool AVX512VL; 		//	AVX512 Vector Length Extensions
		bool AVX512BW; 		//	AVX512 Byte + Word
		bool AVX512DQ; 		//	AVX512 Doubleword + Quadword
		bool AVX512IFMA;	//	AVX512 Integer 52-bit Fused Multiply-Add
		bool AVX512VBMI;	//	AVX512 Vector Byte Manipulation Instructions
	};

	cpuintrinsics getCpuCapabilities();

#endif