# OpenMP convolution kernel Makefile

# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -fopenmp
LDFLAGS = -lm -fopenmp

# Directories
SRC_DIR = .
BIN_DIR = ../../bin

# Source and object files
SRC = $(SRC_DIR)/conv_openmp.c
OBJ = $(SRC:.c=.o)
TARGET = $(BIN_DIR)/conv_openmp

# Make sure bin directory exists
$(shell mkdir -p $(BIN_DIR))

# Default target
all: $(TARGET)

# Link target
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)
	@echo "OpenMP kernel compiled at $(TARGET)"

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJ) $(TARGET)
	@echo "OpenMP kernel clean-up done"
