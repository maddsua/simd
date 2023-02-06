#include <string.h>
#include <cpuid.h>
#include <stdint.h>

#include "cpucapab.hpp"

cpuintrinsics getCpuCapabilities() {

	auto cpuid = [](int* info, int InfoType){
		__cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
	};

	cpuintrinsics flags;
	memset(&flags, 0, sizeof(flags));

	int32_t cpuinfo[4];
	cpuid(cpuinfo, 0);
	int32_t nIds = cpuinfo[0];

	cpuid(cpuinfo, 0x80000000);
	uint32_t nExIds = cpuinfo[0];

	//	Detect Features
	if (nIds >= 0x00000001){
		cpuid(cpuinfo, 0x00000001);

		flags.MMX		= (cpuinfo[3] & ((int32_t)1 << 23)) != 0;
		flags.SSE		= (cpuinfo[3] & ((int32_t)1 << 25)) != 0;
		flags.SSE2		= (cpuinfo[3] & ((int32_t)1 << 26)) != 0;
		flags.SSE3		= (cpuinfo[2] & ((int32_t)1 << 0)) != 0;

		flags.SSSE3		= (cpuinfo[2] & ((int32_t)1 << 9)) != 0;
		flags.SSE41		= (cpuinfo[2] & ((int32_t)1 << 19)) != 0;
		flags.SSE42		= (cpuinfo[2] & ((int32_t)1 << 20)) != 0;
		flags.AES		= (cpuinfo[2] & ((int32_t)1 << 25)) != 0;

		flags.AVX		= (cpuinfo[2] & ((int32_t)1 << 28)) != 0;
		flags.FMA3		= (cpuinfo[2] & ((int32_t)1 << 12)) != 0;

		flags.RDRAND	= (cpuinfo[2] & ((int32_t)1 << 30)) != 0;
	}

	if (nIds >= 0x00000007){
		cpuid(cpuinfo, 0x00000007);

		flags.AVX2			= (cpuinfo[1] & ((int32_t)1 << 5)) != 0;

		flags.BMI1			= (cpuinfo[1] & ((int32_t)1 << 3)) != 0;
		flags.BMI2			= (cpuinfo[1] & ((int32_t)1 << 8)) != 0;
		flags.ADX			= (cpuinfo[1] & ((int32_t)1 << 19)) != 0;
		flags.SHA			= (cpuinfo[1] & ((int32_t)1 << 29)) != 0;
		flags.PREFETCHWT1	= (cpuinfo[2] & ((int32_t)1 << 0)) != 0;

		flags.AVX512F		= (cpuinfo[1] & ((int32_t)1 << 16)) != 0;
		flags.AVX512CD		= (cpuinfo[1] & ((int32_t)1 << 28)) != 0;
		flags.AVX512PF		= (cpuinfo[1] & ((int32_t)1 << 26)) != 0;
		flags.AVX512ER		= (cpuinfo[1] & ((int32_t)1 << 27)) != 0;
		flags.AVX512VL		= (cpuinfo[1] & ((int32_t)1 << 31)) != 0;
		flags.AVX512BW		= (cpuinfo[1] & ((int32_t)1 << 30)) != 0;
		flags.AVX512DQ		= (cpuinfo[1] & ((int32_t)1 << 17)) != 0;
		flags.AVX512IFMA	= (cpuinfo[1] & ((int32_t)1 << 21)) != 0;
		flags.AVX512VBMI	= (cpuinfo[2] & ((int32_t)1 << 1)) != 0;
	}

	if (nExIds >= 0x80000001){
		cpuid(cpuinfo,0x80000001);

		flags.X64	= (cpuinfo[3] & ((int32_t)1 << 29)) != 0;
		flags.ABM	= (cpuinfo[2] & ((int32_t)1 << 5)) != 0;
		flags.SSE4A	= (cpuinfo[2] & ((int32_t)1 << 6)) != 0;
		flags.FMA4	= (cpuinfo[2] & ((int32_t)1 << 16)) != 0;
		flags.XOP	= (cpuinfo[2] & ((int32_t)1 << 11)) != 0;
	}

	return flags;
}