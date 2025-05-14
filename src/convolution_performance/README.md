# Convolution Performance Analysis Framework

This project provides a comprehensive framework for analyzing and comparing the performance of convolution kernel implementations across three different parallel computing paradigms:

1. **Sequential C**
2. **OpenMP (shared-memory parallelism)**
3. **MPI (distributed-memory parallelism)**

## Project Structure

```
convolution_performance/
├── Makefile                   # Main Makefile to compile all implementations
├── benchmarks/                # Configuration files for experiments
│   └── main_config.txt        # Main benchmark configuration
├── bin/                       # Compiled executables 
├── plots/                     # Generated performance plots
├── results/                   # Experimental results
│   ├── parsed_data/           # Aggregated and processed results
│   └── raw_perf_data/         # Raw performance data from perf
├── scripts/                   # Scripts for compilation, running, and analysis
│   ├── aggregate_results.py   # Aggregates raw perf data into CSV
│   ├── compile_kernels.sh     # Compiles all kernel implementations
│   ├── lib_perf_parser.py     # Library for parsing perf output
│   ├── plot_*.py              # Scripts for generating performance plots
│   ├── run_experiment.sh      # Runs experiments defined in benchmark configs
│   └── test_full_pipeline.sh  # End-to-end test script
└── src/                       # Source code for all implementations
    ├── mpi/                   # MPI implementation with Makefile
    ├── openmp/                # OpenMP implementation with Makefile
    └── seq/                   # Sequential implementation with Makefile
```

## Feature Overview

- **Dummy Kernel Implementations**: Configurable dummy convolution kernels for each paradigm that simulate computational workload
- **Performance Data Collection**: Uses Linux `perf` for detailed performance metrics 
- **Flexible Benchmarking**: Configurable benchmark parameters (image size, kernel size, batch size, etc.)
- **Parallel Scaling Analysis**: Compare performance across different thread/process counts
- **Comprehensive Visualization**: Generate plots for:
  - Speedup and parallel efficiency
  - Computational throughput (GFLOPs)
  - Cache misses
  - Instructions per cycle (IPC)
  - Other customizable performance metrics

## Getting Started

### Prerequisites

- GCC compiler with OpenMP support
- MPI implementation (OpenMPI or MPICH)
- Python 3 with pandas, matplotlib, and seaborn
- Linux perf tools

### Basic Usage

1. **Compile all implementations**:
   ```bash
   make all
   ```

2. **Run experiments** with the default configuration:
   ```bash
   ./scripts/run_experiment.sh benchmarks/main_config.txt
   ```

3. **Aggregate results** into a CSV file:
   ```bash
   python ./scripts/aggregate_results.py
   ```

4. **Generate plots**:
   ```bash
   python ./scripts/plot_speedup_efficiency.py
   python ./scripts/plot_throughput.py
   python ./scripts/plot_custom_perf_metrics.py --metrics "instructions,cycles,cache-misses,ipc"
   ```

5. **End-to-end test**:
   ```bash
   ./scripts/test_full_pipeline.sh
   ```

## Configuration

Benchmark configurations are defined in CSV files under the `benchmarks/` directory. Each configuration specifies:

- **KERNEL_TYPE**: Implementation type (SEQ, OPENMP, or MPI)
- **IMG_RES**: Image resolution (e.g., 512x512)
- **KERNEL_SIZE**: Convolution kernel size (e.g., 3x3)
- **BATCH_SIZE**: Input batch size
- **CORE_CONFIG**: 
  - For Sequential: c1 (by convention)
  - For OpenMP: Number of threads (e.g., 2, 4, 8)
  - For MPI: p followed by number of processes (e.g., p2, p4, p8)
- **REPETITIONS**: Number of repetitions for statistical reliability
- **WARMUPS**: Number of warmup runs to prime caches

## Analysis and Visualization

The framework generates various plots to analyze performance:

1. **Speedup and Efficiency**:
   - Per-implementation speedup relative to sequential baseline
   - Parallel efficiency (speedup/cores)
   - Combined comparisons between OpenMP and MPI

2. **Throughput Analysis**:
   - GFLOPs per implementation and core configuration
   - Throughput scaling with core count
   - Data size vs. throughput relationships

3. **Custom Performance Metrics**:
   - Cache usage and miss rates
   - Instruction counts and IPC
   - User-defined metrics from perf data

## Extending the Framework

- **Adding New Kernel Implementations**: Create a new directory under `src/` with your implementation and a Makefile
- **Adding New Performance Metrics**: Modify `lib_perf_parser.py` to extract additional metrics from perf output
- **Custom Visualizations**: Create new plotting scripts based on the existing ones

## License

This project is licensed under the MIT License - see the LICENSE file for details.
