#include <iostream>
#include <array>

#define BLOCK_SIZE	(8)


void print_hex(const uint8_t str[], size_t size) {
	for(size_t i = 0; i < size; i++)
		printf("%02x", str[i]);
}
void print_binary(const uint8_t* data, size_t size) {
	for (size_t m = 0; m < size; m++) {
		for (size_t n = 0; n < 8; n++)
			putc((data[m] & (0b10000000 >> n)) ? '1' : '0', stdout);
		printf(" ");
	}
}

std::array<uint8_t, BLOCK_SIZE> xorbuff(const uint8_t* data, const uint8_t* key) {

	std::array<uint8_t, BLOCK_SIZE> result;
	for (size_t i = 0; i < BLOCK_SIZE; i++)
		result[i] = data[i] ^ key[i];

	return result;
}


int main() {

	std::cout << "This example performs data encoding using the key.\r\nPlease don't confuse it with encryption, XOR ops are only the part of encryption algorithms\r\n\r\n";

	std::array<uint8_t, BLOCK_SIZE> data = {'t', 'e', 's', 't', 'd', 'a', 't', 'a'};

	std::array<uint8_t, BLOCK_SIZE> key = {'k', 'e', 'y', '_', 'h', 'e', 'r', 'e'};

	//	print data
	std::cout << "Data:\r\n";
	print_hex(data.data(), BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(data.data(), BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	print key
	std::cout << "Key:\r\n";
	print_hex(key.data(), BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(key.data(), BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	XOR and print
	auto xored = xorbuff(data.data(), key.data());
	std::cout << "XORed data:\r\n";
	print_hex(xored.data(), BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(xored.data(), BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	XOR XORed data with the ket again to restore it
	auto restored = xorbuff(xored.data(), key.data());
	std::cout << "Restored data:\r\n";
	print_hex(restored.data(), BLOCK_SIZE);
	std::cout << "\r\n";
	print_binary(restored.data(), BLOCK_SIZE);
	std::cout << "\r\n\r\n";

	//	report results
	std::cout << ((restored == data) ? "Data matched" : "ERROR: Data didn't match");

	return 0;
}