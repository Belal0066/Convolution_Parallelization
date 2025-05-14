import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_custom_metrics(csv_filepath, plots_dir, metrics_arg):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Aggregated CSV file not found at {csv_filepath}")
        return

    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Parse metrics argument string into a list of individual metrics
    if isinstance(metrics_arg, str):
        metrics_to_plot = [m.strip() for m in metrics_arg.split(',')]
    else:
        metrics_to_plot = metrics_arg

    for metric_col_base in metrics_to_plot:
        metric_mean_col = f"{metric_col_base}_mean"
        if metric_mean_col not in df.columns:
            print(f"Warning: Metric '{metric_mean_col}' not found in CSV. Skipping plot for it.")
            continue

        # Group by workload for all implementations
        workload_groups = df.groupby(['img_res_x', 'img_res_y', 'kernel_size_x', 'kernel_size_y', 'batch_size'])

        for name, group in workload_groups:
            img_x, img_y, krn_x, krn_y, batch = name
            workload_str = f"Img{img_x}x{img_y}_Krn{krn_x}x{krn_y}_B{batch}"
            
            # Get data for each implementation
            seq_run = group[group['kernel_type'] == 'SEQ']
            omp_data = group[group['kernel_type'] == 'OPENMP'].sort_values(by='num_threads_or_procs')
            mpi_data = group[group['kernel_type'] == 'MPI'].sort_values(by='num_threads_or_procs')
            
            seq_value = None
            if not seq_run.empty and pd.notna(seq_run[metric_mean_col].iloc[0]):
                seq_value = seq_run[metric_mean_col].iloc[0]

            # Plot OpenMP metrics if available
            if not omp_data.empty and not omp_data[metric_mean_col].isnull().all():
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=omp_data, x='num_threads_or_procs', y=metric_mean_col, marker='o', label=f'OpenMP {metric_col_base}')
                
                # Add sequential value for comparison if available
                if seq_value is not None:
                    plt.axhline(seq_value, linestyle=':', color='red', label=f'Sequential ({seq_value:.2e})')
                
                plt.title(f'OpenMP {metric_col_base} vs. Threads\nWorkload: {workload_str}')
                plt.xlabel('Number of Threads')
                plt.ylabel(f'{metric_col_base} (Mean)')
                plt.xticks(omp_data['num_threads_or_procs'].unique())
                plt.legend()
                plt.tight_layout()
                plot_filename = os.path.join(plots_dir, f"{metric_col_base}_openmp_vs_threads_{workload_str}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved OpenMP {metric_col_base} plot to {plot_filename}")
            
            # Plot MPI metrics if available
            if not mpi_data.empty and not mpi_data[metric_mean_col].isnull().all():
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=mpi_data, x='num_threads_or_procs', y=metric_mean_col, marker='o', label=f'MPI {metric_col_base}')
                
                # Add sequential value for comparison if available
                if seq_value is not None:
                    plt.axhline(seq_value, linestyle=':', color='red', label=f'Sequential ({seq_value:.2e})')
                
                plt.title(f'MPI {metric_col_base} vs. Processes\nWorkload: {workload_str}')
                plt.xlabel('Number of Processes')
                plt.ylabel(f'{metric_col_base} (Mean)')
                plt.xticks(mpi_data['num_threads_or_procs'].unique())
                plt.legend()
                plt.tight_layout()
                plot_filename = os.path.join(plots_dir, f"{metric_col_base}_mpi_vs_processes_{workload_str}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved MPI {metric_col_base} plot to {plot_filename}")
                
            # Create combined plot if both implementations have data
            if (not omp_data.empty and not omp_data[metric_mean_col].isnull().all() and 
                not mpi_data.empty and not mpi_data[metric_mean_col].isnull().all()):
                
                plt.figure(figsize=(10, 6))
                
                # Prepare combined dataframe
                omp_plot_data = omp_data.copy()
                omp_plot_data['implementation'] = 'OpenMP'
                mpi_plot_data = mpi_data.copy()
                mpi_plot_data['implementation'] = 'MPI'
                combined_df = pd.concat([omp_plot_data, mpi_plot_data])
                
                # Create plot
                sns.lineplot(data=combined_df, x='num_threads_or_procs', y=metric_mean_col, 
                            hue='implementation', style='implementation', marker='o')
                
                # Add sequential baseline if available
                if seq_value is not None:
                    plt.axhline(seq_value, linestyle=':', color='red', label=f'Sequential ({seq_value:.2e})')
                
                plt.title(f'OpenMP vs MPI {metric_col_base}\nWorkload: {workload_str}')
                plt.xlabel('Number of Threads/Processes')
                plt.ylabel(f'{metric_col_base} (Mean)')
                plt.legend()
                plt.tight_layout()
                plot_filename = os.path.join(plots_dir, f"{metric_col_base}_omp_vs_mpi_{workload_str}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved combined {metric_col_base} plot to {plot_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot custom perf metrics.")
    parser.add_argument("--csv_file", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/results/parsed_data/aggregated_summary.csv", help="Path to aggregated CSV file.")
    parser.add_argument("--plots_dir", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/plots", help="Directory to save plots.")
    parser.add_argument("--metrics", default="ipc,cache_miss_rate,cache-misses,LLC-load-misses", 
                        help="Comma-separated list of base metric names to plot (e.g., 'ipc,cache-misses'). '_mean' will be appended.")
    args = parser.parse_args()
    
    plot_custom_metrics(args.csv_file, args.plots_dir, args.metrics)