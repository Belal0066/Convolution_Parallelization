#!/bin/bash

# Configuration
WARMUP_ITERATIONS=3
MEASUREMENT_ITERATIONS=10
THREAD_COUNTS=(1 2 4 8 16)
IMAGE_SIZES=("512x512" "1024x1024" "2048x2048")
OUTPUT_DIR="benchmark_results"
PERF_DATA_DIR="perf_data"
INPUT_DIR="Inputs"
BIN_DIR="../bin"
MPI_PROCESSES=4  # Number of MPI processes to use

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $PERF_DATA_DIR

# Function to check if input file exists
check_input_file() {
    local size=$1
    local input_file="${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw"
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file $input_file not found"
        return 1
    fi
    return 0
}

# Function to run benchmark for a single configuration
run_benchmark() {
    local impl=$1
    local threads=$2
    local size=$3
    local iter=$4
    local is_warmup=$5
    
    # Parse image size
    IFS='x' read -r width height <<< "$size"
    
    # Check if input file exists
    if ! check_input_file "$size"; then
        return 1
    fi
    
    # Set environment variables
    export OMP_NUM_THREADS=$threads
    
    # Prepare output files
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local perf_file="$PERF_DATA_DIR/${impl}_${threads}_${size}_${timestamp}.perf"
    local time_file="$OUTPUT_DIR/${impl}_${threads}_${size}_${timestamp}.csv"
    
    # Create output files
    touch "$perf_file"
    touch "$time_file"
    
    # Run with perf stat
    if [ "$is_warmup" = true ]; then
        echo "Warmup run $iter for $impl with $threads threads, size $size"
        if [[ $impl == *"mpi"* ]]; then
            perf stat -o "$perf_file" -e cycles,instructions,cache-misses,cache-references,branches,branch-misses \
                mpirun -np $MPI_PROCESSES ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        else
            perf stat -o "$perf_file" -e cycles,instructions,cache-misses,cache-references,branches,branch-misses \
                ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        fi
    else
        echo "Measurement run $iter for $impl with $threads threads, size $size"
        if [[ $impl == *"mpi"* ]]; then
            perf stat -o "$perf_file" -e cycles,instructions,cache-misses,cache-references,branches,branch-misses \
                mpirun -np $MPI_PROCESSES ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        else
            perf stat -o "$perf_file" -e cycles,instructions,cache-misses,cache-references,branches,branch-misses \
                ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        fi
    fi
    
    # Extract timing from perf output if file exists and is not empty
    if [ -s "$perf_file" ]; then
        local time=$(grep "seconds time elapsed" "$perf_file" | awk '{print $1}')
        if [ ! -z "$time" ]; then
            echo "$threads,$size,$time" >> "$time_file"
        fi
    fi
}

# Main benchmarking loop
for impl in "serialized" "mpi_conv" "parallelized"; do
    # Check if executable exists
    if [ ! -f "${BIN_DIR}/${impl}" ]; then
        echo "Error: Executable ${BIN_DIR}/${impl} not found. Skipping..."
        continue
    fi
    
    echo "Benchmarking $impl implementation..."
    
    for size in "${IMAGE_SIZES[@]}"; do
        # Check if input file exists for this size
        if ! check_input_file "$size"; then
            echo "Skipping size $size due to missing input file"
            continue
        fi
        
        # For serialized implementation, only use 1 thread
        if [ "$impl" = "serialized" ]; then
            threads=1
            # Warmup runs
            for ((i=1; i<=$WARMUP_ITERATIONS; i++)); do
                run_benchmark $impl $threads $size $i true
            done
            
            # Measurement runs
            for ((i=1; i<=$MEASUREMENT_ITERATIONS; i++)); do
                run_benchmark $impl $threads $size $i false
            done
        else
            # For parallel implementations, use all thread counts
            for threads in "${THREAD_COUNTS[@]}"; do
                # Warmup runs
                for ((i=1; i<=$WARMUP_ITERATIONS; i++)); do
                    run_benchmark $impl $threads $size $i true
                done
                
                # Measurement runs
                for ((i=1; i<=$MEASUREMENT_ITERATIONS; i++)); do
                    run_benchmark $impl $threads $size $i false
                done
            done
        fi
    done
done

# Generate summary report only if we have data
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    echo "Generating summary report..."
    echo "Implementation,Threads,Size,Average Time (s),Std Dev" > $OUTPUT_DIR/summary.csv

    for impl in "serialized" "mpi_conv" "parallelized"; do
        for size in "${IMAGE_SIZES[@]}"; do
            # For serialized implementation, only process 1 thread
            if [ "$impl" = "serialized" ]; then
                threads=1
                if [ -f "$OUTPUT_DIR/${impl}_${threads}_${size}_"*".csv" ]; then
                    avg=$(awk -F',' '{sum+=$3} END {if(NR>0) print sum/NR; else print "N/A"}' $OUTPUT_DIR/${impl}_${threads}_${size}_*.csv)
                    std=$(awk -F',' -v avg=$avg '{sum+=($3-avg)^2} END {if(NR>0) print sqrt(sum/NR); else print "N/A"}' $OUTPUT_DIR/${impl}_${threads}_${size}_*.csv)
                    echo "$impl,$threads,$size,$avg,$std" >> $OUTPUT_DIR/summary.csv
                fi
            else
                # For parallel implementations, process all thread counts
                for threads in "${THREAD_COUNTS[@]}"; do
                    if [ -f "$OUTPUT_DIR/${impl}_${threads}_${size}_"*".csv" ]; then
                        avg=$(awk -F',' '{sum+=$3} END {if(NR>0) print sum/NR; else print "N/A"}' $OUTPUT_DIR/${impl}_${threads}_${size}_*.csv)
                        std=$(awk -F',' -v avg=$avg '{sum+=($3-avg)^2} END {if(NR>0) print sqrt(sum/NR); else print "N/A"}' $OUTPUT_DIR/${impl}_${threads}_${size}_*.csv)
                        echo "$impl,$threads,$size,$avg,$std" >> $OUTPUT_DIR/summary.csv
                    fi
                done
            fi
        done
    done
else
    echo "No benchmark data collected. Check if input files exist and executables are compiled correctly."
fi

echo "Benchmarking completed. Results are in $OUTPUT_DIR and $PERF_DATA_DIR" 