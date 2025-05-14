# MPI convolution kernel Makefile

# Compiler and flags
CC = mpicc
CFLAGS = -O3 -march=native
LDFLAGS =

# Directories
SRC_DIR = .
BIN_DIR = ../../bin

# Source and object files
SRC = $(SRC_DIR)/conv_mpi.c
OBJ = $(SRC:.c=.o)
TARGET = $(BIN_DIR)/conv_mpi

# Make sure bin directory exists
$(shell mkdir -p $(BIN_DIR))

# Default target
all: $(TARGET)

# Link target
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)
	@echo "MPI kernel compiled at $(TARGET)"

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJ) $(TARGET)
	@echo "MPI kernel clean-up done"
