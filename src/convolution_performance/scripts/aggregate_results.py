import os
import pandas as pd
from lib_perf_parser import parse_perf_file
import argparse

def aggregate_data(raw_data_dir, output_csv_path):
    all_data = []
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".perf.txt"):
            filepath = os.path.join(raw_data_dir, filename)
            print(f"Parsing {filename}...")
            
            # Parse filename for parameters
            parts = filename.replace(".perf.txt", "").split('_')
            if len(parts) < 6:
                print(f"Warning: Could not parse filename {filename}. Skipping.")
                continue

            kernel_type = parts[0].upper()
            img_res = parts[1]
            kernel_sz_str = parts[2]
            batch_size_str = parts[3] # e.g., b1
            core_config_str = parts[4] # e.g., c1 or c4
            # rep_num_str = parts[5] # e.g., rep1

            try:
                img_w, img_h = map(int, img_res.split('x'))
                krn_w, krn_h = map(int, kernel_sz_str.split('x'))
                batch_size = int(batch_size_str[1:]) # remove 'b'
                
                core_config_val = core_config_str # Store as is for grouping, e.g. "c1", "c4", "p4"
                num_cores_or_threads = 1 # Default for SEQ or if parsing fails
                if kernel_type == "OPENMP":
                    num_cores_or_threads = int(core_config_str[1:]) # remove 'c'
                elif kernel_type == "SEQ":
                    num_cores_or_threads = 1
                elif kernel_type == "MPI":
                    num_cores_or_threads = int(core_config_str[1:]) # remove 'p'
                
            except ValueError as e:
                print(f"Warning: Error parsing parameters from filename {filename}: {e}. Skipping.")
                continue

            perf_metrics = parse_perf_file(filepath)
            if perf_metrics:
                row = {
                    'kernel_type': kernel_type,
                    'img_res_x': img_w,
                    'img_res_y': img_h,
                    'kernel_size_x': krn_w,
                    'kernel_size_y': krn_h,
                    'batch_size': batch_size,
                    'core_config_raw': core_config_str, # Raw string like c1, c4 for grouping
                    'num_threads_or_procs': num_cores_or_threads # Parsed numeric value
                }
                row.update(perf_metrics)
                all_data.append(row)
            else:
                print(f"Warning: No metrics parsed for {filename}")

    if not all_data:
        print("No data parsed. Exiting.")
        return

    df = pd.DataFrame(all_data)

    # Define grouping keys
    group_keys = ['kernel_type', 'img_res_x', 'img_res_y', 'kernel_size_x', 'kernel_size_y', 
                  'batch_size', 'core_config_raw', 'num_threads_or_procs']
    
    # Columns to aggregate
    metrics_to_aggregate = [col for col in df.columns if col not in group_keys and df[col].dtype in ['int64', 'float64']]
    
    agg_funcs = {metric: ['mean', 'std'] for metric in metrics_to_aggregate}
    
    aggregated_df = df.groupby(group_keys, as_index=False).agg(agg_funcs)
    
    # Flatten MultiIndex columns
    aggregated_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in aggregated_df.columns.values]
    aggregated_df.rename(columns=lambda x: x.replace('_val', ''), inplace=True) # if as_index was True
    
    # Calculate derived metrics (mean only for simplicity, std could be propagated)
    if 'instructions_mean' in aggregated_df.columns and 'cycles_mean' in aggregated_df.columns:
        aggregated_df['ipc_mean'] = aggregated_df['instructions_mean'] / aggregated_df['cycles_mean']
    
    if 'cache-misses_mean' in aggregated_df.columns and 'cache-references_mean' in aggregated_df.columns:
        # Avoid division by zero if cache-references is 0 or None
        aggregated_df['cache_miss_rate_mean'] = aggregated_df.apply(
            lambda row: row['cache-misses_mean'] / row['cache-references_mean'] if row['cache-references_mean'] and row['cache-references_mean'] > 0 else None, axis=1
        )

    # Rename exec_time_s_mean to mean_exec_time_s for clarity
    if 'exec_time_s_mean' in aggregated_df.columns:
        aggregated_df.rename(columns={'exec_time_s_mean': 'mean_exec_time_s',
                                      'exec_time_s_std': 'std_exec_time_s'}, inplace=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    aggregated_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"Aggregated data saved to {output_csv_path}")
    print("\nAggregated DataFrame head:")
    print(aggregated_df.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate perf results.")
    parser.add_argument("--raw_dir", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/results/raw_perf_data", help="Directory with raw perf files.")
    parser.add_argument("--output_csv", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/results/parsed_data/aggregated_summary.csv", help="Path to save aggregated CSV.")
    args = parser.parse_args()
    
    aggregate_data(args.raw_dir, args.output_csv)