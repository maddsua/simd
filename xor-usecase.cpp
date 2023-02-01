#include <iostream>
#include <array>

#include "workbench.hpp"

#define XOR_BLOCK_SIZE	(8)


std::array<uint8_t, XOR_BLOCK_SIZE> xorbuff(const uint8_t* data, const uint8_t* key) {

	std::array<uint8_t, XOR_BLOCK_SIZE> result;
	for (size_t i = 0; i < XOR_BLOCK_SIZE; i++)
		result[i] = data[i] ^ key[i];

	return result;
}


int main() {

	std::cout << "This example performs data encoding using the key.\r\nPlease don't confuse it with encryption, XOR ops are only the part of encryption algorithms\r\n\r\n";

	std::array<uint8_t, XOR_BLOCK_SIZE> data = {'t', 'e', 's', 't', 'd', 'a', 't', 'a'};

	std::array<uint8_t, XOR_BLOCK_SIZE> key = {'k', 'e', 'y', '_', 'h', 'e', 'r', 'e'};

	//	print data
	std::cout << "Data:\r\n";
	print_hex(data.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(data.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	print key
	std::cout << "Key:\r\n";
	print_hex(key.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(key.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	XOR and print
	auto xored = xorbuff(data.data(), key.data());

	std::cout << "XORed data:\r\n";
	print_hex(xored.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(xored.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	XOR XORed data with the key again to restore it
	auto restored = xorbuff(xored.data(), key.data());

	std::cout << "Restored data:\r\n";
	print_hex(restored.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(restored.data(), XOR_BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	report results
	std::cout << ((restored == data) ? "Data matched" : "ERROR: Data didn't match") << "\r\n";

	return 0;
}
