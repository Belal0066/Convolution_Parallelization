#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <benchmark_config_file>"
    exit 1
fi

BENCHMARK_CONFIG_FILE="$1"
BASE_DIR=$(dirname "$0")/..
BIN_DIR="$BASE_DIR/bin"
RAW_PERF_DIR="$BASE_DIR/results/raw_perf_data"

mkdir -p "$RAW_PERF_DIR"

# Define perf events - make this configurable later if needed
PERF_EVENTS="task-clock,cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses,branch-instructions,branch-misses"

echo "Starting experiments based on $BENCHMARK_CONFIG_FILE..."

# Skip header line
tail -n +2 "$BENCHMARK_CONFIG_FILE" | while IFS=',' read -r KERNEL_TYPE IMG_RES KERNEL_SIZE BATCH_SIZE CORE_CONFIG REPETITIONS WARMUPS; do
    # Trim whitespace
    KERNEL_TYPE=$(echo "$KERNEL_TYPE" | xargs)
    IMG_RES=$(echo "$IMG_RES" | xargs)
    KERNEL_SIZE=$(echo "$KERNEL_SIZE" | xargs)
    BATCH_SIZE=$(echo "$BATCH_SIZE" | xargs)
    CORE_CONFIG=$(echo "$CORE_CONFIG" | xargs)
    REPETITIONS=$(echo "$REPETITIONS" | xargs)
    WARMUPS=$(echo "$WARMUPS" | xargs)

    echo "Processing: $KERNEL_TYPE, IMG: $IMG_RES, KRN: $KERNEL_SIZE, BATCH: $BATCH_SIZE, CORES: $CORE_CONFIG, REPS: $REPETITIONS, WARMUPS: $WARMUPS"

    IMG_WIDTH=$(echo "$IMG_RES" | cut -d'x' -f1)
    IMG_HEIGHT=$(echo "$IMG_RES" | cut -d'x' -f2)
    KERNEL_WIDTH=$(echo "$KERNEL_SIZE" | cut -d'x' -f1)
    KERNEL_HEIGHT=$(echo "$KERNEL_SIZE" | cut -d'x' -f2)

    EXECUTABLE=""
    CORE_CONFIG_FILENAME_PART="" # For filename consistency

    # Command arguments are the same for all implementations
    CMD_ARGS="$IMG_WIDTH $IMG_HEIGHT $KERNEL_WIDTH $KERNEL_HEIGHT $BATCH_SIZE"
    
    case "$KERNEL_TYPE" in
        SEQ)
            EXECUTABLE="$BIN_DIR/conv_seq"
            CORE_CONFIG_FILENAME_PART="c1" # By convention for sequential
            RUN_CMD="$EXECUTABLE $CMD_ARGS"
            ;;
        OPENMP)
            EXECUTABLE="$BIN_DIR/conv_openmp"
            export OMP_NUM_THREADS="$CORE_CONFIG" # CORE_CONFIG is num_threads for OpenMP
            CORE_CONFIG_FILENAME_PART="c${CORE_CONFIG}"
            RUN_CMD="$EXECUTABLE $CMD_ARGS"
            ;;
        MPI)
            EXECUTABLE="$BIN_DIR/conv_mpi"
            # For MPI, CORE_CONFIG should be formatted like p4 for 4 processes
            NUM_PROCESSES="${CORE_CONFIG#p}" # Remove 'p' prefix
            export OMP_NUM_THREADS=1 # Ensure no implicit OpenMP threading in MPI processes
            CORE_CONFIG_FILENAME_PART="${CORE_CONFIG}" # e.g., p4
            RUN_CMD="mpirun -np $NUM_PROCESSES $EXECUTABLE $CMD_ARGS"
            ;;
        *)
            echo "Unknown KERNEL_TYPE: $KERNEL_TYPE. Skipping."
            continue
            ;;
    esac

    if [ ! -f "$EXECUTABLE" ]; then
        echo "Executable $EXECUTABLE not found. Please compile first. Skipping."
        continue
    fi

    CMD_ARGS="$IMG_WIDTH $IMG_HEIGHT $KERNEL_WIDTH $KERNEL_HEIGHT $BATCH_SIZE $CORE_CONFIG" # Last arg is informational for kernel

    # Warm-up runs
    echo "  Performing $WARMUPS warm-up runs..."
    for ((w=1; w<=WARMUPS; w++)); do
        eval "$RUN_CMD" > /dev/null 2>&1
    done

    # Measurement runs
    echo "  Performing $REPETITIONS measurement runs..."
    for ((r=1; r<=REPETITIONS; r++)); do
        FILENAME_BASE="${KERNEL_TYPE,,}_${IMG_RES}_${KERNEL_SIZE}_b${BATCH_SIZE}_${CORE_CONFIG_FILENAME_PART}_rep${r}"
        PERF_OUTPUT_FILE="$RAW_PERF_DIR/${FILENAME_BASE}.perf.txt"
        
        echo "    Running rep $r, output to $PERF_OUTPUT_FILE"
        
        # Using eval to handle command construction with perf
        # The '--' separates perf options from the command and its arguments
        eval "perf stat -e $PERF_EVENTS -o $PERF_OUTPUT_FILE -- $RUN_CMD" > /dev/null 2>&1
        
        # Check if perf output file was created and is not empty
        if [ ! -s "$PERF_OUTPUT_FILE" ]; then
            echo "WARNING: perf output file $PERF_OUTPUT_FILE is empty or was not created. Perf might have failed."
            # Attempt to capture stderr from perf directly if the file is empty
            eval "perf stat -e $PERF_EVENTS -- $RUN_CMD" 2> "$PERF_OUTPUT_FILE.error"
            echo "Perf stderr captured in $PERF_OUTPUT_FILE.error"
        fi
        sleep 0.1 # Small delay between runs
    done
    echo "-----------------------------------------------------"
done

echo "All experiments finished."