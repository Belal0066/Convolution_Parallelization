# Convolution Benchmarking Suite

This suite provides tools for benchmarking and analyzing the performance of different convolution implementations.

## Prerequisites

- Linux operating system
- Python 3.x with required packages:
  ```bash
  pip install pandas matplotlib seaborn
  ```
- perf tool installed:
  ```bash
  sudo apt-get install linux-tools-common linux-tools-generic
  ```
- MPI and OpenMP development tools:
  ```bash
  sudo apt-get install openmpi-bin libopenmpi-dev
  ```

## Directory Structure

```
.
├── Makefile              # Main Makefile for compilation
├── benchmark.sh          # Main benchmarking script
├── analyze_benchmarks.py # Analysis and visualization script
├── src/                  # Source code directory
│   ├── openmp-mpi/      # OpenMP and MPI implementations
│   └── mpi/             # Pure MPI implementation
├── benchmark_results/    # Generated timing data
├── perf_data/           # Generated perf stat data
└── benchmark_plots/     # Generated visualization plots
```

## Compilation

The suite includes three implementations:
1. Serial implementation
2. Pure MPI implementation
3. Hybrid OpenMP-MPI implementation

To compile all implementations:
```bash
make
```

To clean build artifacts:
```bash
make clean
```

## Usage

1. **Prepare Input Data**
   - Place your input images in the working directory
   - Name them according to the pattern: `input_<size>.raw`
   - Example: `input_512x512.raw`, `input_1024x1024.raw`, etc.

2. **Run Benchmarks**
   ```bash
   chmod +x benchmark.sh
   ./benchmark.sh
   ```
   This will:
   - Compile all implementations
   - Run warm-up iterations
   - Perform measurement runs
   - Collect perf stat data
   - Generate timing data

3. **Analyze Results**
   ```bash
   python3 analyze_benchmarks.py
   ```
   This will generate:
   - Timing comparison plots
   - Speedup analysis
   - Performance metric visualizations
   - Summary statistics

## Output Files

### Timing Data
- CSV files in `benchmark_results/`
- Format: `implementation_threads_size_timestamp.csv`
- Contains: thread count, image size, execution time

### Performance Data
- Perf stat output in `perf_data/`
- Format: `implementation_threads_size_timestamp.perf`
- Contains: cycles, instructions, cache misses, etc.

### Visualizations
- PNG files in `benchmark_plots/`
- Includes:
  - Timing vs Thread Count
  - Speedup Analysis
  - Cache Performance
  - Branch Prediction Efficiency

### Summary Statistics
- JSON file: `benchmark_plots/summary_stats.json`
- Contains aggregated statistics for all metrics

## Customization

You can modify the following parameters in `benchmark.sh`:
- `WARMUP_ITERATIONS`: Number of warm-up runs
- `MEASUREMENT_ITERATIONS`: Number of measurement runs
- `THREAD_COUNTS`: Array of thread counts to test
- `IMAGE_SIZES`: Array of image sizes to test
- `MPI_PROCESSES`: Number of MPI processes to use

## Notes

- The benchmarking script assumes the executables are in the current directory
- Make sure all executables are compiled with appropriate optimization flags
- For MPI runs, the script uses the number of processes specified in `MPI_PROCESSES`
- Results may vary based on system load and configuration
- The hybrid implementation uses both MPI and OpenMP, so thread count affects both levels of parallelism 