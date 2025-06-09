# Accelerating Convolution Kernels using Parallel Programming

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-green.svg)]()
[![Language](https://img.shields.io/badge/Language-C%2FC%2B%2B%7CCUDA-orange.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Linux%7CUnix-lightgrey.svg)]()

> **A comprehensive research project analyzing and optimizing image convolution algorithms through multiple parallel computing paradigms including OpenMP, MPI, and CUDA implementations.**

## Overview

This research project presents a systematic analysis of parallel image convolution implementations, benchmarking performance across multiple computational paradigms. Developed by students from **Ain Shams University's Faculty of Engineering**, our work demonstrates significant performance improvements through strategic parallelization, achieving up to **11.7x speedup** with CUDA GPU acceleration and **4.9x speedup** with MPI distributed computing.

### Key Contributions

- **Multi-paradigm Implementation**: Serial, OpenMP, MPI, and CUDA implementations of Gaussian blur convolution
- **Performance Analysis**: Strong and weak scaling studies with detailed hardware profiling
- **Benchmarking**: Automated testing framework with statistical rigor
- **Theoretical Validation**: Empirical results compared against Amdahl's Law predictions


## Performance Highlights

| Implementation | Max Threads | Peak GFLOPS | Memory BW (MB/s) | Efficiency | Speedup |
|----------------|-------------|-------------|------------------|------------|---------|
| Serial         | 1           | 2.31        | 513.4           | 100%       | 1.0x    |
| OpenMP+MPI     | 16          | 3.88        | 861.7           | 85.2%      | 2.35x   |
| MPI            | 16          | 11.41       | 2535.5          | 92.8%      | 4.9x    |
| CUDA RTX 3050  | 2048        | 18.5        | 4200.0          | 89.0%      | 11.7x   |

*Performance measured on AMD Ryzen 9 5900HS + NVIDIA RTX 3050 (6GB GDDR6)*

## 📄 Complete Research Report

For detailed analysis, methodology, and comprehensive results, see our complete research paper:

**[📖 Parallel Convolution Research Report (PDF)](doc/final_report/Parallel_report.pdf)**

*The report includes extensive performance analysis, theoretical foundations, implementation details, and visual results with charts and graphs from our experimental evaluation.*

## Quick Start

### Prerequisites

```bash
# System dependencies
sudo apt-get update
sudo apt-get install build-essential gcc mpi mpich-dev libomp-dev

# Optional: CUDA Toolkit for GPU implementation
sudo apt-get install nvidia-cuda-toolkit

# Python dependencies for analysis
pip install -r performance_analysis/requirements.txt
```

### Build All Implementations

```bash
# Build all parallel implementations
make all

# Verify build
ls bin/
# Output: mpi_conv parallelized serialized
```

### Run Quick Benchmark

```bash
cd performance_analysis
./benchmark.sh -i all
```

### Generate Performance Analysis

```bash
python visualize_benchmark.py
```

## Implementation Details

### 1. Serial Implementation (`src/openmp-mpi/serialized.c`)
- **Purpose**: Baseline performance reference
- **Algorithm**: Standard 2D convolution with Gaussian blur kernel
- **Complexity**: O(n²) for n×n image
- **Memory**: Single-threaded, cache-optimized access patterns

### 2. OpenMP Implementation (`src/openmp-mpi/parallelized.c`)
- **Parallelization Strategy**: Loop-level parallelism with work-sharing
- **Thread Management**: Dynamic scheduling with optimal chunk sizes
- **Memory Model**: Shared memory with false-sharing mitigation
- **Scalability**: Up to 16 threads on tested hardware

### 3. MPI Implementation (`src/mpi/mpi_conv.c`)
- **Decomposition**: 2D block decomposition with halo exchange
- **Communication**: Optimized MPI_Sendrecv for boundary data
- **Load Balancing**: Static partitioning with consideration for edge effects
- **Fault Tolerance**: Graceful degradation on process failure

### 4. CUDA Implementation (`src/cuda/cuda_convolute.cu`)
- **Kernel Design**: 2D thread blocks optimized for RTX 3050 architecture
- **Memory Hierarchy**: Coalesced global memory access, shared memory caching
- **Performance**: 11.7x speedup over serial implementation (3100ms → 265ms)
- **Hardware Target**: NVIDIA RTX 3050 with 2048 CUDA cores, 6GB GDDR6
- **Profiling**: Integrated with NVIDIA Nsight Compute for detailed analysis

## Benchmarking Framework

### Performance Metrics Collection

Our benchmarking framework captures comprehensive performance data:

- **Execution Metrics**: Wall-clock time, CPU cycles, instructions executed
- **Memory Metrics**: L1/L2/LLC cache misses, memory bandwidth utilization
- **Computational Metrics**: GFLOPS, IPC (Instructions Per Cycle)
- **Parallel Metrics**: Speedup, efficiency, scalability analysis

### Statistical Rigor

- **Warmup Iterations**: 3 iterations to stabilize cache state
- **Measurement Iterations**: 10 iterations for statistical significance
- **Outlier Detection**: Automated removal of statistical outliers
- **Confidence Intervals**: 95% confidence intervals for all measurements

### Hardware Profiling

Integration with Linux `perf` subsystem provides detailed insights:

```bash
# Example profiling command
perf stat -e cycles,instructions,cache-misses,LLC-misses \
  mpirun -np 8 ./bin/mpi_conv input.raw 2048 2048 1
```

## Research Results

### CUDA GPU Performance Analysis

Our CUDA implementation achieved exceptional performance on the NVIDIA RTX 3050:

- **11.7x Speedup**: From 3100ms (CPU) to 265ms (GPU) for 1920×2520 images
- **89% Compute Throughput**: Near-optimal utilization of GPU resources  
- **High Cache Efficiency**: 96% L1 and 98% L2 cache hit rates
- **Memory Bound Characteristics**: 26% memory throughput suggests compute-bound workload

### Strong Scaling Analysis

Our strong scaling analysis demonstrates excellent parallel efficiency:

- **MPI Implementation**: Peak 4.9x speedup at 4 processes, good scalability up to 16 processes
- **Hybrid OpenMP+MPI**: 2.35x maximum speedup, limited by hybrid overheads at fixed problem sizes
- **Theoretical Validation**: Results closely follow Amdahl's Law predictions for serial fraction limitations

### Weak Scaling Analysis

Weak scaling results show superior performance characteristics:

- **Hybrid Approach**: Near-ideal weak scaling with constant execution time (~0.15-0.18s)
- **MPI-Only**: Good scalability but higher execution time (~0.20-0.25s)  
- **Memory Efficiency**: Hybrid approach maintains higher bandwidth utilization (861 MB/s vs 2535 MB/s)

### Hardware Platform

Our evaluation was conducted on research-grade hardware:

**CPU Platform:**
- **Processor**: AMD Ryzen™ 9 5900HS (8 cores × 2 threads = 16 logical threads)  
- **Clock Speed**: Up to 4.68 GHz boost
- **Cache Hierarchy**: 512 KB L1, 4 MB L2, 16 MB L3 (shared)
- **Memory**: 16 GB DDR4 @ 3200 MHz, dual-channel

**GPU Platform:**
- **GPU**: NVIDIA GeForce RTX 3050 Laptop (6 GB GDDR6)
- **CUDA Cores**: 2048 cores at ~1740 MHz boost clock
- **Architecture**: Ampere with RT cores and Tensor cores
- **Memory Bandwidth**: 192 GB/s theoretical peak

**Software Environment:**
- **OS**: Ubuntu 24.04.2 LTS (Kernel 6.11.0)
- **Compiler**: GNU GCC 13.3.0 with OpenMP support
- **MPI**: OpenMPI with optimized communication primitives
- **CUDA**: Toolkit 12.9 with Nsight Compute 2025.2.0

### Performance Bottlenecks Identified

1. **Memory Bandwidth**: Primary bottleneck for large images (>4K×4K)
2. **Cache Locality**: L1 miss rate increases with problem size  
3. **Communication**: MPI halo exchange overhead at high process counts
4. **Load Imbalance**: Edge effects in 2D decomposition
5. **GPU Memory Coalescing**: 94% of L2 requests uncoalesced in CUDA implementation
6. **Hybrid Overheads**: Combined MPI+OpenMP overheads limit strong scaling performance

## Project Structure

```
Convolution_Parallelization/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── Makefile                     # Build configuration
├── bin/                         # Compiled binaries
│   ├── serialized              # Serial implementation
│   ├── parallelized            # OpenMP implementation
│   └── mpi_conv                # MPI implementation
├── src/                         # Source code
│   ├── main.cpp                # Common utilities
│   ├── openmp-mpi/             # OpenMP implementations
│   ├── mpi/                    # MPI implementation
│   └── cuda/                   # CUDA GPU implementation
├── performance_analysis/        # Benchmarking framework
│   ├── benchmark.sh            # Automated benchmarking
│   ├── visualize_benchmark.py  # Analysis and visualization
│   ├── README.md               # Detailed framework documentation
│   ├── benchmark_data/         # Performance results
│   ├── plots/                  # Generated visualizations
│   └── Inputs/                 # Test datasets
└── doc/                        # Research documentation
    ├── final_report/           # Academic paper and results
    └── proposal/               # Project proposal
```

## Test Datasets

We provide a comprehensive suite of standardized test images designed for reproducible benchmarking across different computational loads and memory requirements. All datasets are located in `performance_analysis/Inputs/` and follow a consistent raw grayscale format for optimal performance measurement.

### Available Test Images

| Resolution | File Size  
|------------|-----------
| 512×512    | 256 KB    
| 1024×1024  | 1 MB     
| 2048×2048  | 4 MB      
| 4096×4096  | 16 MB     
| 8192×8192  | 64 MB     
| 1920×2520  | 4.8 MB    

### Dataset Characteristics

**Image Format:**
- **Type**: Raw grayscale (8-bit unsigned integers)
- **Source**: High-quality waterfall image optimized for convolution testing
- **Encoding**: Uncompressed binary format for consistent I/O performance
- **Byte Order**: Little-endian for x86-64 compatibility

**Scaling Strategy:**
- **Geometric Progression**: Powers of 2 scaling (512² → 8192²) for cache analysis
- **Real-world Dimension**: 1920×2520 represents typical high-resolution photography
- **Memory Footprint**: Ranges from 256 KB to 64 MB for comprehensive testing

### Dataset Generation

The test suite includes a Python utility (`inputImgResize.py`) for generating additional test datasets:

```python
# Generate custom resolution datasets
python performance_analysis/Inputs/inputImgResize.py

# Creates standardized test images from base waterfall image
# Output: input_blur_waterfall_grey_{width}x{height}.raw
```

**Key Features:**
- **High-Quality Resampling**: Uses Lanczos interpolation for optimal image quality
- **Consistent Naming**: Standardized filename convention for automated benchmarking
- **Batch Processing**: Generates multiple resolutions in a single execution
- **Memory Efficient**: Streams data to minimize RAM usage during generation

### Usage in Benchmarking

**Automated Selection:**
The benchmarking framework automatically selects appropriate test images based on:
- Available system memory
- Target thread/process count
- Scaling analysis type (strong vs. weak scaling)

**Manual Selection:**
```bash
# Test specific resolution
./bin/serialized performance_analysis/Inputs/input_blur_waterfall_grey_2048x2048.raw 2048 2048 10

# CUDA testing with real-world dimensions
./src/cuda/cuda_conv performance_analysis/Inputs/waterfall_grey_1920_2520.raw 1920 2520 100 grey

# MPI scaling test
mpirun -np 8 ./bin/mpi_conv performance_analysis/Inputs/input_blur_waterfall_grey_4096x4096.raw 4096 4096 5
```

### Performance Scaling Characteristics

**Strong Scaling Tests:**
- Use fixed image size (typically 8192×8192) with varying thread counts
- Tests parallel efficiency under constant computational load
- Reveals Amdahl's Law limitations and communication overheads

**Weak Scaling Tests:**
- Scale image size proportionally with thread count
- Maintains constant work per processing element
- Evaluates memory bandwidth and cache hierarchy performance




## Contributing

We welcome contributions from the research community:

1. **Fork** the repository *(it's not stealing if we encourage it)*
2. **Create** a feature branch (`git checkout -b feature/FeatureGamda`)
3. **Commit** your changes (`git commit -m 'Add FeatureGamda'`)
4. **Push** to the branch (`git push origin feature/FeatureGamda`)
5. **Open** a Pull Request *(we promise to review it faster than our code compiles)*

### Contribution Guidelines

- Follow existing code style and formatting
- Include comprehensive tests for new features
- Update documentation
- Ensure backward compatibility
- Enjoy your time :D

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{parallel_convolution_asu_2025,
  title={Parallel Image Convolution: A Comprehensive Performance Analysis},
  author={Belal Anas Awad and Mohamed Salah Fathy and Salma Mohamed Youssef and Salma Hisham Hassan Wagdy},
  institution={Faculty of Engineering, Ain Shams University},
  address={Cairo, Egypt},
  year={2025},
  howpublished={\url{https://github.com/Belal0066/Convolution_Parallelization}},
  note={Research project for CSE355: Parallel \& Distributed Algorithms}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Ain Shams University** - Faculty of Engineering, Computer and Systems Engineering Department
- **Dr. Islam Tharwat Abdel Halim** - Research Supervisor and Course Instructor  
- **Eng. Hassan Ahmed** - Technical Guidance and Mentorship
- **CSE355** - Parallel & Distributed Algorithms Course
- **Research Community** - Open source tools and libraries that made this work possible

## Contact & Support

**Research Team - Ain Shams University:**
- **Mohamed Salah Fathy** - 21p0117@eng.asu.edu.eg  
- **Salma Mohamed Youssef** - 21p0148@eng.asu.edu.eg
- **Salma Hisham Hassan Wagdy** - 21P0124@eng.asu.edu.eg
- **Belal Anas Awad** - 21p0072@eng.asu.edu.eg

---

**Keywords**: Parallel Computing, Image Processing, OpenMP, MPI, CUDA, Performance Analysis, High Performance Computing, Convolution, Computer Vision
