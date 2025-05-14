import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_throughput(csv_filepath, plots_dir):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Aggregated CSV file not found at {csv_filepath}")
        return

    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if 'mean_exec_time_s' not in df.columns:
        print("Error: 'mean_exec_time_s' not in CSV. Cannot calculate throughput.")
        return

    # Calculate GFLOPs (example, adjust based on actual convolution)
    # Assuming 2 FLOPs per multiply-accumulate (MAC)
    # For a convolution: img_h * img_w * krn_h * krn_w * batch_size * 2 operations
    df['gflops_val'] = (df['img_res_x'] * df['img_res_y'] * \
                       df['kernel_size_x'] * df['kernel_size_y'] * \
                       df['batch_size'] * 2) / (df['mean_exec_time_s'] * 1e9)
    
    df['data_size_pixels'] = df['img_res_x'] * df['img_res_y'] * df['batch_size']


    # Group by workload for all implementations
    workload_groups = df.groupby(['img_res_x', 'img_res_y', 'kernel_size_x', 'kernel_size_y', 'batch_size'])

    for name, group in workload_groups:
        img_x, img_y, krn_x, krn_y, batch = name
        workload_str = f"Img{img_x}x{img_y}_Krn{krn_x}x{krn_y}_B{batch}"

        # Get data for each implementation
        seq_run = group[group['kernel_type'] == 'SEQ']
        omp_df = group[group['kernel_type'] == 'OPENMP'].copy().sort_values(by='num_threads_or_procs')
        mpi_df = group[group['kernel_type'] == 'MPI'].copy().sort_values(by='num_threads_or_procs')

        # Plot OpenMP throughput if available
        if not omp_df.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=omp_df, x='num_threads_or_procs', y='gflops_val', marker='o', label='OpenMP GFLOPs')
            
            # Add sequential throughput for comparison if available
            if not seq_run.empty and pd.notna(seq_run['gflops_val'].iloc[0]):
                plt.axhline(seq_run['gflops_val'].iloc[0], linestyle=':', color='red', 
                           label=f'Sequential GFLOPs ({seq_run["gflops_val"].iloc[0]:.2f})')

            plt.title(f'OpenMP Throughput (GFLOPs) vs. Threads\nWorkload: {workload_str}')
            plt.xlabel('Number of Threads')
            plt.ylabel('GFLOPs')
            plt.tight_layout()
            plt.legend()
            plot_filename = os.path.join(plots_dir, f"throughput_openmp_vs_threads_{workload_str}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved OpenMP throughput plot to {plot_filename}")

        # Plot MPI throughput if available
        if not mpi_df.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=mpi_df, x='num_threads_or_procs', y='gflops_val', marker='o', label='MPI GFLOPs')
            
            # Add sequential throughput for comparison if available
            if not seq_run.empty and pd.notna(seq_run['gflops_val'].iloc[0]):
                plt.axhline(seq_run['gflops_val'].iloc[0], linestyle=':', color='red', 
                           label=f'Sequential GFLOPs ({seq_run["gflops_val"].iloc[0]:.2f})')

            plt.title(f'MPI Throughput (GFLOPs) vs. Processes\nWorkload: {workload_str}')
            plt.xlabel('Number of Processes')
            plt.ylabel('GFLOPs')
            plt.tight_layout()
            plt.legend()
            plot_filename = os.path.join(plots_dir, f"throughput_mpi_vs_processes_{workload_str}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved MPI throughput plot to {plot_filename}")
            
        # Create combined throughput plot for OpenMP vs MPI if both available
        if not omp_df.empty and not mpi_df.empty:
            plt.figure(figsize=(10, 6))
            
            # Prepare combined dataframe for plotting
            omp_df_plot = omp_df.copy()
            omp_df_plot['implementation'] = 'OpenMP'
            mpi_df_plot = mpi_df.copy()
            mpi_df_plot['implementation'] = 'MPI'
            combined_df = pd.concat([omp_df_plot, mpi_df_plot])
            
            # Create the plot
            sns.lineplot(data=combined_df, x='num_threads_or_procs', y='gflops_val', 
                        hue='implementation', style='implementation', marker='o')
            
            # Add sequential throughput
            if not seq_run.empty and pd.notna(seq_run['gflops_val'].iloc[0]):
                plt.axhline(seq_run['gflops_val'].iloc[0], linestyle=':', color='red', 
                           label=f'Sequential GFLOPs ({seq_run["gflops_val"].iloc[0]:.2f})')

            plt.title(f'OpenMP vs MPI Throughput (GFLOPs)\nWorkload: {workload_str}')
            plt.xlabel('Number of Threads/Processes')
            plt.ylabel('GFLOPs')
            plt.tight_layout()
            plt.legend()
            plot_filename = os.path.join(plots_dir, f"throughput_omp_vs_mpi_{workload_str}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved combined throughput plot to {plot_filename}")
        plt.ylabel('Throughput (GFLOPs)')
        plt.xticks(group['num_threads_or_procs'].unique())
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, f"throughput_openmp_vs_threads_{workload_str}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved throughput plot to {plot_filename}")

    # Throughput vs Data Size (comparing best of each implementation: Sequential, OpenMP, MPI)
    # Group by data size and find the best configuration for each implementation
    omp_df = df[df['kernel_type'] == 'OPENMP'].copy()
    mpi_df = df[df['kernel_type'] == 'MPI'].copy()
    seq_df = df[df['kernel_type'] == 'SEQ'].copy()
    
    best_throughput_data = []
    
    # Find best OpenMP configuration for each data size
    if not omp_df.empty:
        best_omp_indices = omp_df.loc[omp_df.groupby('data_size_pixels')['gflops_val'].idxmax()]
        best_throughput_data.append(best_omp_indices[['data_size_pixels', 'gflops_val', 'kernel_type']])
    
    # Find best MPI configuration for each data size
    if not mpi_df.empty:
        best_mpi_indices = mpi_df.loc[mpi_df.groupby('data_size_pixels')['gflops_val'].idxmax()]
        best_throughput_data.append(best_mpi_indices[['data_size_pixels', 'gflops_val', 'kernel_type']])
    
    # Add sequential data
    if not seq_df.empty:
        best_throughput_data.append(seq_df[['data_size_pixels', 'gflops_val', 'kernel_type']])
    
    # Combine all data
    if best_throughput_data:
        combined_throughput_df = pd.concat(best_throughput_data).sort_values(by='data_size_pixels')
        
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=combined_throughput_df, x='data_size_pixels', y='gflops_val', hue='kernel_type', marker='o', style='kernel_type')
        plt.title('Throughput (GFLOPs) vs. Data Size - Best Configuration Per Implementation')
        plt.xlabel('Data Size (Total Input Pixels)')
        plt.ylabel('Throughput (GFLOPs)')
        plt.xscale('log', base=2) # Often data sizes vary exponentially
        plt.legend(title='Implementation')
        plt.tight_layout()
        plot_filename_datasize = os.path.join(plots_dir, f"throughput_vs_datasize_comparison.png")
        plt.savefig(plot_filename_datasize)
        plt.close()
        print(f"Saved throughput vs data size comparison plot to {plot_filename_datasize}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot throughput.")
    parser.add_argument("--csv_file", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/results/parsed_data/aggregated_summary.csv", help="Path to aggregated CSV file.")
    parser.add_argument("--plots_dir", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/plots", help="Directory to save plots.")
    args = parser.parse_args()
    
    plot_throughput(args.csv_file, args.plots_dir)