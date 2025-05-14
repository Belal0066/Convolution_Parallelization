# Makefile for the convolution performance project
# Main Makefile that calls sub-Makefiles in different implementation directories

.PHONY: all clean seq openmp mpi

# Default target builds all implementations
all: seq openmp mpi

# Sequential implementation
seq:
	$(MAKE) -C src/seq

# OpenMP implementation
openmp:
	$(MAKE) -C src/openmp

# MPI implementation
mpi:
	$(MAKE) -C src/mpi

# Clean all build artifacts
clean:
	$(MAKE) -C src/seq clean
	$(MAKE) -C src/openmp clean
	$(MAKE) -C src/mpi clean
