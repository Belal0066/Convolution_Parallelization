#!/bin/bash
set -euo pipefail

# Test script to validate the full experimental pipeline with all three implementations
BASE_DIR=$(dirname "$0")/..
BENCHMARK_FILE="$BASE_DIR/benchmarks/test_config.txt"
RESULTS_DIR="$BASE_DIR/results"
PLOTS_DIR="$BASE_DIR/plots/test_run"

echo "Starting end-to-end test of the experimental pipeline..."

# Create a small test configuration file
echo "# KERNEL_TYPE, IMG_RES, KERNEL_SIZE, BATCH_SIZE, CORE_CONFIG, REPETITIONS, WARMUPS" > "$BENCHMARK_FILE"
echo "SEQ,128x128,3x3,1,c1,2,1" >> "$BENCHMARK_FILE"
echo "OPENMP,128x128,3x3,1,2,2,1" >> "$BENCHMARK_FILE"
echo "MPI,128x128,3x3,1,p2,2,1" >> "$BENCHMARK_FILE"

echo "Created test benchmark configuration at $BENCHMARK_FILE"

# Step 1: Compile all kernels using Makefiles
echo "Step 1: Compiling kernels using Makefiles..."
make -C "$BASE_DIR" all

# Step 2: Run experiments
echo "Step 2: Running experiments..."
"$BASE_DIR/scripts/run_experiment.sh" "$BENCHMARK_FILE"

# Step 3: Aggregate results
echo "Step 3: Aggregating results..."
python3 "$BASE_DIR/scripts/aggregate_results.py" --raw_dir "$RESULTS_DIR/raw_perf_data" --output_csv "$RESULTS_DIR/parsed_data/test_results.csv"

# Step 4: Generate plots
echo "Step 4: Generating plots..."
mkdir -p "$PLOTS_DIR"
python3 "$BASE_DIR/scripts/plot_speedup_efficiency.py" --csv_file "$RESULTS_DIR/parsed_data/test_results.csv" --plots_dir "$PLOTS_DIR"
python3 "$BASE_DIR/scripts/plot_throughput.py" --csv_file "$RESULTS_DIR/parsed_data/test_results.csv" --plots_dir "$PLOTS_DIR"
python3 "$BASE_DIR/scripts/plot_custom_perf_metrics.py" --csv_file "$RESULTS_DIR/parsed_data/test_results.csv" --plots_dir "$PLOTS_DIR" --metrics "instructions,cycles,cache-misses,ipc"

echo "End-to-end test completed successfully!"
echo "Results saved to:"
echo "- Raw data: $RESULTS_DIR/raw_perf_data"
echo "- Parsed data: $RESULTS_DIR/parsed_data/test_results.csv"
echo "- Plots: $PLOTS_DIR"
