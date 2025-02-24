#include <iostream>
#include <vector>
#include <chrono>
#include <smmintrin.h> // SSE4.1
#include <immintrin.h> // AVX

void add_no_SSE(long size, std::vector<int>& first_array, const std::vector<int>& second_array) {
    for (long i = 0; i < size; ++i) {
        first_array[i] += second_array[i];
    }
}

void add_SSE(long size, std::vector<int>& first_array, const std::vector<int>& second_array) {
    long i = 0;
    for (; i + 4 <= size; i += 4) {
        __m128i first_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&first_array[i]));
        __m128i second_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&second_array[i]));

        first_values = _mm_add_epi32(first_values, second_values);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&first_array[i]), first_values);
    }
    // Handle leftovers
    for (; i < size; ++i) {
        first_array[i] += second_array[i];
    }
}
// changes made here
int main() {
    constexpr long size = 67108864;
    std::vector<int> array1(size, 1);
    std::vector<int> array2(size, 2);

    auto start = std::chrono::high_resolution_clock::now();
    add_no_SSE(size, array1, array2);
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count();
    double Mflops = 2e-6 * size / seconds;
    std::cout << "Performance in Mflops: " << Mflops << " Mflop/s" << std::endl;

    // Reset the arrays for a fair comparison
    std::fill(array1.begin(), array1.end(), 1);
    std::fill(array2.begin(), array2.end(), 2);

    start = std::chrono::high_resolution_clock::now();
    add_SSE(size, array1, array2);
    end = std::chrono::high_resolution_clock::now();

    seconds = std::chrono::duration<double>(end - start).count();
    double Gflops = 2e-9 * size / seconds;
    std::cout << "Performance of SSE in Gflops: " << Gflops << " Gflop/s" << std::endl;

    return 0;
}
