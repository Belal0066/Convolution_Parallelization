# Sequential convolution kernel Makefile

# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native
LDFLAGS =

# Directories
SRC_DIR = .
BIN_DIR = ../../bin

# Source and object files
SRC = $(SRC_DIR)/conv_seq.c
OBJ = $(SRC:.c=.o)
TARGET = $(BIN_DIR)/conv_seq

# Make sure bin directory exists
$(shell mkdir -p $(BIN_DIR))

# Default target
all: $(TARGET)

# Link target
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)
	@echo "Sequential kernel compiled at $(TARGET)"

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJ) $(TARGET)
	@echo "Sequential kernel clean-up done"
