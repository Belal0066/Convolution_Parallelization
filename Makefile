CC = gcc
CFLAGS = -O3 -Wall -Wextra
MPICC = mpicc
MPICFLAGS = -O3 -Wall -Wextra
OPENMP_FLAGS = -openmp
BIN_DIR = bin

all: $(BIN_DIR) serialized parallelized mpi_conv

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

serialized: src/openmp-mpi/serialized.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/serialized src/openmp-mpi/serialized.c

parallelized: src/openmp-mpi/parallelized.c | $(BIN_DIR)
	$(MPICC) $(MPICFLAGS) $(OPENMP_FLAGS) -o $(BIN_DIR)/parallelized src/openmp-mpi/parallelized.c

mpi_conv: src/mpi/mpi_conv.c | $(BIN_DIR)
	$(MPICC) $(MPICFLAGS) -o $(BIN_DIR)/mpi_conv src/mpi/mpi_conv.c

clean_compiled:
	rm -f $(BIN_DIR)/serialized $(BIN_DIR)/parallelized $(BIN_DIR)/mpi_conv

clean_benchmark:
	rm -rf benchmark_results perf_data benchmark_plots

clean_all: clean_compiled clean_benchmark

.PHONY: all clean_compiled clean_benchmark clean_all
