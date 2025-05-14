#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h> // For usleep
#include <time.h>   // For clock

// Dummy function to simulate work
void perform_dummy_convolution_omp(long img_pixels, long kernel_pixels, int batch_size, int num_threads) {
    volatile double computation_sum = 0; // Use a reduction to avoid false sharing if threads wrote to a shared var
    long total_ops_per_thread_chunk = (img_pixels * kernel_pixels * batch_size * 20) / (1000 * num_threads) ; 
    if (total_ops_per_thread_chunk == 0) total_ops_per_thread_chunk = 1;

    #pragma omp parallel for reduction(+:computation_sum)
    for (int t = 0; t < num_threads * 10; ++t) { // Spread work over more iterations than threads
        volatile double computation = 0;
        for (long i = 0; i < total_ops_per_thread_chunk; ++i) {
            computation += (double)i * 0.00001;
            computation /= 1.00001;
        }
        computation_sum += computation;
    }
    // usleep((useconds_t)((img_pixels * kernel_pixels * batch_size * 20) / (10000 * num_threads) ));
}


int main(int argc, char *argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <img_width> <img_height> <kernel_width> <kernel_height> <batch_size> <num_threads_info>\n", argv[0]);
        return 1;
    }

    long img_width = atol(argv[1]);
    long img_height = atol(argv[2]);
    long kernel_width = atol(argv[3]);
    long kernel_height = atol(argv[4]);
    int batch_size = atoi(argv[5]);
    // int num_threads_arg = atoi(argv[6]); // For info, OMP_NUM_THREADS controls actual threads

    int num_threads = omp_get_max_threads(); // Get actual number of threads used

    long img_pixels = img_width * img_height;
    long kernel_pixels = kernel_width * kernel_height;
    
    clock_t start_time, end_time;

    start_time = clock();
    perform_dummy_convolution_omp(img_pixels, kernel_pixels, batch_size, num_threads);
    end_time = clock();
    
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; // in ms
    // printf("OpenMP Kernel (Threads: %d) execution time: %.3f ms\n", num_threads, cpu_time_used);
    
    return 0;
}