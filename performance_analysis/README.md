# Performance Analysis

This directory contains comprehensive performance analysis tools and results for the convolution parallelization project. It includes benchmarking scripts, visualization tools, test data, and performance metrics for different parallel implementations.

## Directory Structure

```
performance_analysis/
├── benchmark.sh              # Main benchmarking script
├── visualize_benchmark.py    # Visualization and analysis script
├── requirements.txt          # Python dependencies
├── blur_serial              # Serial implementation binary
├── benchmark_data/          # Benchmark results and logs
│   ├── all_benchmarks_strong.csv
│   ├── all_benchmarks_weak.csv
│   ├── summary_strong.csv
│   ├── summary_weak.csv
│   ├── benchmark.log
│   └── perf_data/          # Performance profiling data
├── Inputs/                 # Test input images
│   ├── input_blur_waterfall_grey_*.raw
│   ├── inputImgResize.py
│   └── waterfall_grey_1920_2520.raw
├── plots/                  # Generated performance plots
│   ├── strong/            # Strong scaling analysis plots
│   └── weak/              # Weak scaling analysis plots
└── venvPy/                # Python virtual environment
```

## Features

### Benchmarking (`benchmark.sh`)
- **Automated Performance Testing**: Comprehensive benchmarking of three implementations:
  - `serialized`: Single-threaded OpenMP implementation
  - `parallelized`: Multi-threaded OpenMP implementation
  - `mpi_conv`: MPI-based parallel implementation

- **Scaling Analysis**:
  - **Strong Scaling**: Fixed problem size with varying thread counts (1, 2, 4, 8, 12, 16)
  - **Weak Scaling**: Proportional problem size increase with thread count

- **Performance Metrics Collection**:
  - Execution time and standard deviation
  - CPU cycles and instructions
  - Cache miss rates (L1 and LLC)
  - GFLOPS (Giga Floating-Point Operations Per Second)
  - Memory bandwidth (MB/s)
  - Instructions Per Cycle (IPC)

- **Hardware Profiling**: Uses `perf` for detailed performance counter analysis

### Visualization (`visualize_benchmark.py`)
- **Comprehensive Plot Generation**:
  - Strong and weak scaling efficiency
  - Performance metrics vs. thread count
  - 3D surface plots for multi-dimensional analysis
  - Amdahl's Law theoretical vs. actual speedup comparison

- **Supported Metrics**:
  - Execution time
  - GFLOPS performance
  - Memory bandwidth utilization
  - Cache miss rates
  - Instructions per cycle (IPC)

## Usage

### Prerequisites
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure compiled binaries are available in `../bin/`:
   - `serialized`
   - `parallelized` 
   - `mpi_conv`

### Running Benchmarks

#### Run All Implementations
```bash
./benchmark.sh
```

#### Run Specific Implementation
```bash
./benchmark.sh -i serialized    # Serial implementation only
./benchmark.sh -i parallelized  # OpenMP implementation only
./benchmark.sh -i mpi_conv      # MPI implementation only
```

#### Help
```bash
./benchmark.sh -h
```

### Generate Visualizations
```bash
python visualize_benchmark.py
```

This will generate plots in the `plots/` directory for both strong and weak scaling analysis.

## Test Data

### Input Images
- Multiple resolution test images (512×512 to 8192×8192)
- Raw grayscale format for consistent benchmarking
- `inputImgResize.py`: Utility for generating different image sizes

### Image Sizes Tested
- 512×512 pixels
- 1024×1024 pixels  
- 2048×2048 pixels
- 4096×4096 pixels
- 8192×8192 pixels

## Benchmark Configuration

### Thread Counts
- 1, 2, 4, 8, 12, 16 threads (matches hardware capabilities)

### Iterations
- **Warmup**: 3 iterations (excluded from measurements)
- **Measurement**: 10 iterations for statistical accuracy

### MPI Configuration
- Base process count: 8 processes
- Optimized for available hardware resources

## Output Files

### CSV Data Files
- `summary_strong.csv`: Strong scaling benchmark summary
- `summary_weak.csv`: Weak scaling benchmark summary  
- `all_benchmarks_strong.csv`: Detailed strong scaling results
- `all_benchmarks_weak.csv`: Detailed weak scaling results

### Performance Plots
- **Strong Scaling**:
  - Efficiency vs. threads
  - GFLOPS vs. threads
  - Memory bandwidth vs. threads
  - Cache miss rates vs. threads
  - IPC vs. threads
  - Amdahl's Law comparison

- **Weak Scaling**:
  - Efficiency analysis
  - 3D surface plots for multi-metric visualization

### Logs
- `benchmark.log`: Detailed execution logs and debugging information

## Performance Metrics

### Primary Metrics
- **Execution Time**: Wall-clock time for convolution operation
- **GFLOPS**: Computational throughput
- **Memory Bandwidth**: Data transfer rate (MB/s)
- **Efficiency**: Parallel efficiency calculation

### Hardware Counters
- **CPU Cycles**: Total cycles consumed
- **Instructions**: Total instructions executed
- **L1 Cache Misses**: Level 1 cache miss count
- **LLC Misses**: Last Level Cache miss count
- **IPC**: Instructions per cycle ratio

## Analysis Results

The performance analysis reveals:
- Scalability characteristics of different parallel approaches
- Bottlenecks in memory bandwidth and cache utilization
- Optimal thread counts for different problem sizes
- Comparison with theoretical performance limits (Amdahl's Law)

## Dependencies

### System Requirements
- Linux/Unix environment
- GCC compiler with OpenMP support
- MPI implementation (OpenMPI/MPICH)
- `perf` profiling tools

### Python Dependencies
See `requirements.txt`:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- scipy: Scientific computing
- pillow: Image processing utilities

## Notes

- Benchmarks are designed for multi-core systems with at least 16 hardware threads
- Performance results are hardware-dependent
- Use consistent system load conditions for reproducible results
- Virtual environment (`venvPy/`) isolates Python dependencies
