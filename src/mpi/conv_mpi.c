#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h> // For usleep
#include <time.h>   // For clock

// Dummy function to simulate work across MPI processes
void perform_dummy_convolution_mpi(long img_pixels, long kernel_pixels, int batch_size, int rank, int num_procs) {
    volatile double computation = 0;
    
    // Calculate workload per process - distribute work evenly
    long total_ops = img_pixels * kernel_pixels * batch_size * 20; // Arbitrary scaling factor
    long ops_per_proc = total_ops / (1000 * num_procs); // Similar scaling as in openmp
    if (ops_per_proc == 0) ops_per_proc = 1;
    
    // Each process performs its share of the work
    for (long i = 0; i < ops_per_proc; ++i) {
        computation += (double)i * 0.00001;
        computation /= 1.00001;
    }
    
    // Simulate a gather/reduce operation (optional)
    double global_sum = 0;
    MPI_Reduce(&computation, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // To ensure the result is used (prevent optimization)
    if (rank == 0) {
        volatile double result = global_sum;
    }
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    double start_time, end_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Check arguments
    if (argc < 6) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <img_width> <img_height> <kernel_width> <kernel_height> <batch_size> [num_processes_info]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long img_width = atol(argv[1]);
    long img_height = atol(argv[2]);
    long kernel_width = atol(argv[3]);
    long kernel_height = atol(argv[4]);
    int batch_size = atoi(argv[5]);
    // num_procs is already available from MPI_Comm_size
    
    long img_pixels = img_width * img_height;
    long kernel_pixels = kernel_width * kernel_height;
    
    // Use MPI timing
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    start_time = MPI_Wtime();
    
    // Call the convolution function
    perform_dummy_convolution_mpi(img_pixels, kernel_pixels, batch_size, rank, num_procs);
    
    // Synchronize and get end time
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Only rank 0 reports time (not used by perf, but useful for validation)
    if (rank == 0) {
        double exec_time_ms = (end_time - start_time) * 1000.0; // Convert to ms
        // Uncomment for debug output
        // printf("MPI Kernel execution time: %.3f ms\n", exec_time_ms);
    }
    
    MPI_Finalize();
    return 0;
}
