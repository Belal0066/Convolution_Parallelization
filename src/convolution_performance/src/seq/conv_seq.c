#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For usleep
#include <time.h>   // For clock

// Dummy function to simulate work
void perform_dummy_convolution(long img_pixels, long kernel_pixels, int batch_size) {
    volatile double computation = 0;
    long total_ops = img_pixels * kernel_pixels * batch_size * 20; // Arbitrary scaling factor

    // Simulate computation; adjust loop iterations for desired runtime
    for (long i = 0; i < total_ops / 1000; ++i) {
        computation += (double)i * 0.00001;
        computation /= 1.00001;
    }
    // usleep((useconds_t)(total_ops / 10000)); // Alternative: use usleep for rough timing
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <img_width> <img_height> <kernel_width> <kernel_height> <batch_size>\n", argv[0]);
        return 1;
    }

    long img_width = atol(argv[1]);
    long img_height = atol(argv[2]);
    long kernel_width = atol(argv[3]);
    long kernel_height = atol(argv[4]);
    int batch_size = atoi(argv[5]);

    long img_pixels = img_width * img_height;
    long kernel_pixels = kernel_width * kernel_height;

    clock_t start_time, end_time;
    
    start_time = clock();
    perform_dummy_convolution(img_pixels, kernel_pixels, batch_size);
    end_time = clock();

    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; // in ms
    // This time is for basic validation, perf will be the primary source
    // printf("Sequential Kernel execution time: %.3f ms\n", cpu_time_used);

    return 0;
}