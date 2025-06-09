#!/bin/bash

# Configuration
WARMUP_ITERATIONS=3
MEASUREMENT_ITERATIONS=10
THREAD_COUNTS=(1 2 4 8 12 16)  # Matches hardware thread limit
IMAGE_SIZES=("512x512" "1024x1024" "2048x2048" "4096x4096" "8192x8192")
OUTPUT_DIR="benchmark_data"
PERF_DATA_DIR="benchmark_data/perf_data"
INPUT_DIR="Inputs"
BIN_DIR="../bin"
MPI_PROCESSES=8  # Reduced base MPI processes to better match hardware
LOG_FILE="$OUTPUT_DIR/benchmark.log"

# Parse command line arguments
usage() {
    echo "Usage: $0 [-i implementation]"
    echo "Options:"
    echo "  -i implementation    Specify implementation to run (serialized, mpi_conv, parallelized, or all)"
    echo "  -h                  Display this help message"
    exit 1
}

# Default to running all implementations
IMPLEMENTATION="all"

while getopts "i:h" opt; do
    case $opt in
        i)
            case $OPTARG in
                serialized|mpi_conv|parallelized|all)
                    IMPLEMENTATION=$OPTARG
                    ;;
                *)
                    echo "Error: Invalid implementation. Must be one of: serialized, mpi_conv, parallelized, or all"
                    usage
                    ;;
            esac
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $PERF_DATA_DIR

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if input file exists
check_input_file() {
    local size=$1
    local input_file="${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw"
    if [ ! -f "$input_file" ]; then
        log_message "Error: Input file $input_file not found"
        return 1
    fi
    
    # Check file size and available memory
    local file_size
    if command -v stat >/dev/null 2>&1; then
        if stat -f %z "$input_file" >/dev/null 2>&1; then
            file_size=$(stat -f %z "$input_file")
        else
            file_size=$(stat -c %s "$input_file")
        fi
    else
        file_size=$(wc -c < "$input_file")
    fi
    
    if [ -z "$file_size" ]; then
        log_message "Warning: Could not determine file size for $input_file"
        return 0
    fi
    
    local available_mem=$(free -b | awk '/^Mem:/{print $7}')
    
    # For 8192x8192 images, we need at least 4x the file size in memory
    if [[ "$size" == "8192x8192" ]] && [ $((file_size * 4)) -gt $available_mem ]; then
        log_message "Warning: Available memory ($available_mem bytes) might be insufficient for size $size"
    fi
    
    return 0
}

# Function to calculate performance metrics
calculate_metrics() {
    local width=$1
    local height=$2
    local time=$3
    local cycles=$4
    local instructions=$5
    
    # Calculate GFLOPS/s
    local flops=$((2 * width * height * 9))  # 9 operations per pixel
    local gflops=$(echo "scale=6; $flops / ($time * 1000000000)" | bc)
    
    # Calculate bandwidth (MB/s)
    local bytes=$((width * height * 4))  # 4 bytes per pixel
    local bandwidth=$(echo "scale=6; $bytes / ($time * 1000000)" | bc)
    
    # Calculate IPC
    local ipc=$(echo "scale=6; $instructions / $cycles" | bc)
    
    echo "$gflops,$bandwidth,$ipc"
}

# Function to run benchmark for a single configuration
run_benchmark() {
    local impl=$1
    local threads=$2
    local size=$3
    local iter=$4
    local is_warmup=$5
    local scaling_type=$6
    
    # Parse image size
    IFS='x' read -r width height <<< "$size"
    
    # Check if input file exists
    if ! check_input_file "$size"; then
        return 1
    fi
    
    # Adjust MPI processes based on image size
    local mpi_procs=$MPI_PROCESSES
    if [[ "$size" == "8192x8192" ]]; then
        mpi_procs=16  # Use maximum available threads for largest size
    elif [[ "$size" == "4096x4096" ]]; then
        mpi_procs=12  # Use more processes for large size
    elif [[ "$size" == "2048x2048" ]]; then
        mpi_procs=8   # Use 8 processes for medium size
    fi
    
    # Ensure we don't exceed hardware thread limit
    if [ $mpi_procs -gt 16 ]; then
        mpi_procs=16
        log_message "Warning: MPI processes capped at 16 to match hardware limit"
    fi

    # For parallelized implementation
    if [[ "$impl" == "parallelized" ]]; then
        # Use even fewer MPI processes to reduce overhead
        if [[ "$size" == "8192x8192" ]]; then
            mpi_procs=2  # Use 2 MPI processes with 8 OpenMP threads each
            export OMP_NUM_THREADS=8
        elif [[ "$size" == "4096x4096" ]]; then
            mpi_procs=2  # Use 2 MPI processes with 6 OpenMP threads each
            export OMP_NUM_THREADS=6
        elif [[ "$size" == "2048x2048" ]]; then
            mpi_procs=2  # Use 2 MPI processes with 4 OpenMP threads each
            export OMP_NUM_THREADS=4
        elif [[ "$size" == "1024x1024" ]]; then
            mpi_procs=2  # Use 2 MPI processes with 2 OpenMP threads each
            export OMP_NUM_THREADS=2
        elif [[ "$size" == "512x512" ]]; then
            mpi_procs=2  # Use 2 MPI processes with 1 OpenMP thread each
            export OMP_NUM_THREADS=1
        fi
    else
        # For other implementations, use the original thread count
        export OMP_NUM_THREADS=$threads
    fi
    
    # Prepare output files
    local perf_file="$PERF_DATA_DIR/${impl}_${threads}_${size}_${iter}_${scaling_type}.perf"
    local all_benchmarks_file="$OUTPUT_DIR/all_benchmarks_${scaling_type}.csv"
    
    # Create or append to all_benchmarks.csv with headers if it doesn't exist
    if [ ! -f "$all_benchmarks_file" ]; then
        echo "implementation,threads,image_size,width,height,time,cycles,instructions,L1_dcache_load_misses,LLC_load_misses,GFLOPS,bw_MBps,IPC,type" > "$all_benchmarks_file"
    fi
    
    # Run with perf stat
    if [ "$is_warmup" = true ]; then
        log_message "Warmup run $iter for $impl with $threads threads, size $size ($scaling_type scaling)"
        if [[ $impl == *"mpi"* ]]; then
            perf stat -o "$perf_file" -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \
                mpirun -np $mpi_procs ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        else
            perf stat -o "$perf_file" -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \
                ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        fi
    else
        log_message "Measurement run $iter for $impl with $threads threads, size $size ($scaling_type scaling)"
        if [[ $impl == *"mpi"* ]]; then
            perf stat -o "$perf_file" -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \
                mpirun -np $mpi_procs ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        else
            perf stat -o "$perf_file" -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \
                ${BIN_DIR}/$impl ${INPUT_DIR}/input_blur_waterfall_grey_${size}.raw $width $height 1 grey 2>/dev/null || true
        fi
    fi
    
    # Extract metrics from perf output if file exists and is not empty
    if [ -s "$perf_file" ]; then
        local time=$(grep "seconds time elapsed" "$perf_file" | awk '{print $1}')
        local cycles=$(grep "cycles" "$perf_file" | head -n1 | awk '{print $1}' | tr -d ',')
        local instructions=$(grep "instructions" "$perf_file" | awk '{print $1}' | tr -d ',')
        local l1_misses=$(grep "L1-dcache-load-misses" "$perf_file" | awk '{print $1}' | tr -d ',')
        local llc_misses=$(grep "LLC-load-misses" "$perf_file" | awk '{print $1}' | tr -d ',')
        
        # Handle LLC misses that might be "<not counted>"
        if [[ "$llc_misses" == *"<not"* ]]; then
            llc_misses="0"  # Set to 0 if not counted
        fi
        
        if [ ! -z "$time" ] && [ ! -z "$cycles" ] && [ ! -z "$instructions" ]; then
            # Calculate derived metrics
            local metrics=$(calculate_metrics $width $height $time $cycles $instructions)
            IFS=',' read -r gflops bandwidth ipc <<< "$metrics"
            
            # Determine implementation type
            local impl_type="serial"
            if [[ $impl == *"mpi"* ]]; then
                impl_type="mpi"
            elif [[ $impl == *"parallelized"* ]]; then
                impl_type="omp"
            fi
            
            # Append to all_benchmarks.csv
            echo "$impl,$threads,$size,$width,$height,$time,$cycles,$instructions,$l1_misses,$llc_misses,$gflops,$bandwidth,$ipc,$impl_type" >> "$all_benchmarks_file"
        fi
    fi
}

# Function to run strong scaling benchmarks
run_strong_scaling() {
    local impl=$1
    local max_size="8192x8192"  # Fixed workload for strong scaling
    
    log_message "Starting strong scaling benchmarks for $impl with size $max_size"
    
    # For serialized implementation, only use 1 thread
    if [ "$impl" = "serialized" ]; then
        threads=1
        # Warmup runs
        for ((i=1; i<=$WARMUP_ITERATIONS; i++)); do
            run_benchmark $impl $threads $max_size $i true "strong"
        done
        
        # Measurement runs
        for ((i=1; i<=$MEASUREMENT_ITERATIONS; i++)); do
            run_benchmark $impl $threads $max_size $i false "strong"
        done
    else
        # For parallel implementations, skip 1 thread and start from 2
        for threads in "${THREAD_COUNTS[@]:1}"; do
            # Warmup runs
            for ((i=1; i<=$WARMUP_ITERATIONS; i++)); do
                run_benchmark $impl $threads $max_size $i true "strong"
            done
            
            # Measurement runs
            for ((i=1; i<=$MEASUREMENT_ITERATIONS; i++)); do
                run_benchmark $impl $threads $max_size $i false "strong"
            done
        done
    fi
}

# Function to run weak scaling benchmarks
run_weak_scaling() {
    local impl=$1
    
    log_message "Starting weak scaling benchmarks for $impl"
    
    # Create thread-size pairs for weak scaling
    declare -A thread_size_pairs=(
        ["2"]="512x512"
        ["4"]="1024x1024"
        ["8"]="2048x2048"
        ["12"]="4096x4096"
        ["16"]="8192x8192"
    )
    
    # For serialized implementation, run with all image sizes
    if [ "$impl" = "serialized" ]; then
        threads=1
        # Run for each image size
        for size in "${IMAGE_SIZES[@]}"; do
            # Warmup runs
            for ((i=1; i<=$WARMUP_ITERATIONS; i++)); do
                run_benchmark $impl $threads $size $i true "weak"
            done
            
            # Measurement runs
            for ((i=1; i<=$MEASUREMENT_ITERATIONS; i++)); do
                run_benchmark $impl $threads $size $i false "weak"
            done
        done
    else
        # For parallel implementations, use thread-size pairs
        for threads in "${!thread_size_pairs[@]}"; do
            size="${thread_size_pairs[$threads]}"
            # Warmup runs
            for ((i=1; i<=$WARMUP_ITERATIONS; i++)); do
                run_benchmark $impl $threads $size $i true "weak"
            done
            
            # Measurement runs
            for ((i=1; i<=$MEASUREMENT_ITERATIONS; i++)); do
                run_benchmark $impl $threads $size $i false "weak"
            done
        done
    fi
}

# Main benchmarking loop
if [ "$IMPLEMENTATION" = "all" ]; then
    IMPLEMENTATIONS=("serialized" "mpi_conv" "parallelized")
else
    IMPLEMENTATIONS=("$IMPLEMENTATION")
fi

for impl in "${IMPLEMENTATIONS[@]}"; do
    # Check if executable exists
    if [ ! -f "${BIN_DIR}/${impl}" ]; then
        echo "Error: Executable ${BIN_DIR}/${impl} not found. Skipping..."
        continue
    fi
    
    echo "Benchmarking $impl implementation..."
    
    # Run strong scaling benchmarks
    run_strong_scaling $impl
    
    # Run weak scaling benchmarks
    run_weak_scaling $impl
done

# Generate summary reports for both scaling types
for scaling_type in "strong" "weak"; do
    if [ -f "$OUTPUT_DIR/all_benchmarks_${scaling_type}.csv" ]; then
        echo "Generating summary report for ${scaling_type} scaling..."
        echo "implementation,threads,image_size,width,height,type,avg_time,std_time,avg_cycles,std_cycles,avg_instructions,std_instructions,avg_L1_misses,std_L1_misses,avg_LLC_misses,std_LLC_misses,avg_GFLOPS,std_GFLOPS,avg_bw_MBps,std_bw_MBps,avg_IPC,std_IPC" > "$OUTPUT_DIR/summary_${scaling_type}.csv"
        
        # Process the all_benchmarks.csv to generate summary statistics
        awk -F',' '
        NR>1 {
            # Create a unique key based on the relevant fields, including the "type"
            key = $1 "," $2 "," $3 "," $4 "," $5 "," $14  # implementation,threads,image_size,width,height,type
            
            # Accumulate time and square of time for standard deviation
            time[key] += $6
            time2[key] += $6*$6
            
            # Accumulate cycles and square of cycles for standard deviation
            cycles[key] += $7
            cycles2[key] += $7*$7
            
            # Accumulate instructions and square of instructions for standard deviation
            instructions[key] += $8
            instructions2[key] += $8*$8
            
            # Accumulate L1 misses and square of L1 misses for standard deviation
            l1_misses[key] += $9
            l1_misses2[key] += $9*$9
            
            # Accumulate LLC misses and square of LLC misses for standard deviation
            llc_misses[key] += $10
            llc_misses2[key] += $10*$10
            
            # Accumulate GFLOPS and square of GFLOPS for standard deviation
            gflops[key] += $11
            gflops2[key] += $11*$11
            
            # Accumulate bandwidth and square of bandwidth for standard deviation
            bw[key] += $12
            bw2[key] += $12*$12
            
            # Accumulate IPC and square of IPC for standard deviation
            ipc[key] += $13
            ipc2[key] += $13*$13
            
            # Count the number of entries for each unique key
            count[key]++
        }
        END {
            # Output the results for each unique key (combination of all fields)
            for (k in count) {
                n = count[k]
                
                # Calculate averages and standard deviations
                avg_time = time[k]/n
                std_time = sqrt((time2[k]/n) - (avg_time*avg_time))
                
                avg_cycles = cycles[k]/n
                std_cycles = sqrt((cycles2[k]/n) - (avg_cycles*avg_cycles))
                
                avg_instructions = instructions[k]/n
                std_instructions = sqrt((instructions2[k]/n) - (avg_instructions*avg_instructions))
                
                avg_l1_misses = l1_misses[k]/n
                std_l1_misses = sqrt((l1_misses2[k]/n) - (avg_l1_misses*avg_l1_misses))
                
                avg_llc_misses = llc_misses[k]/n
                std_llc_misses = sqrt((llc_misses2[k]/n) - (avg_llc_misses*avg_llc_misses))
                
                avg_gflops = gflops[k]/n
                std_gflops = sqrt((gflops2[k]/n) - (avg_gflops*avg_gflops))
                
                avg_bw = bw[k]/n
                std_bw = sqrt((bw2[k]/n) - (avg_bw*avg_bw))
                
                avg_ipc = ipc[k]/n
                std_ipc = sqrt((ipc2[k]/n) - (avg_ipc*avg_ipc))
                
                # Print out the summary for this key
                print k "," avg_time "," std_time "," avg_cycles "," std_cycles "," avg_instructions "," std_instructions "," avg_l1_misses "," std_l1_misses "," avg_llc_misses "," std_llc_misses "," avg_gflops "," std_gflops "," avg_bw "," std_bw "," avg_ipc "," std_ipc
            }
        }' "$OUTPUT_DIR/all_benchmarks_${scaling_type}.csv" >> "$OUTPUT_DIR/summary_${scaling_type}.csv"
    else
        echo "No benchmark data collected for ${scaling_type} scaling. Check if input files exist and executables are compiled correctly."
    fi
done

echo "Benchmarking completed. Results are in $OUTPUT_DIR"
