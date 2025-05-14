import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_speedup_efficiency(csv_filepath, plots_dir):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Aggregated CSV file not found at {csv_filepath}")
        return

    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Ensure 'mean_exec_time_s' and 'num_threads_or_procs' exist
    if 'mean_exec_time_s' not in df.columns or 'num_threads_or_procs' not in df.columns:
        print("Error: Required columns 'mean_exec_time_s' or 'num_threads_or_procs' not in CSV.")
        return

    # Group by workload (img_res, kernel_size, batch_size)
    workload_groups = df.groupby(['img_res_x', 'img_res_y', 'kernel_size_x', 'kernel_size_y', 'batch_size'])

    for name, group in workload_groups:
        img_x, img_y, krn_x, krn_y, batch = name
        workload_str = f"Img{img_x}x{img_y}_Krn{krn_x}x{krn_y}_B{batch}"
        
        seq_runs = group[group['kernel_type'] == 'SEQ']
        omp_runs = group[group['kernel_type'] == 'OPENMP'].sort_values(by='num_threads_or_procs')
        mpi_runs = group[group['kernel_type'] == 'MPI'].sort_values(by='num_threads_or_procs')

        if seq_runs.empty:
            print(f"Warning: No sequential (SEQ) data found for workload {workload_str}. Skipping speedup plot.")
            continue
        
        baseline_time = seq_runs['mean_exec_time_s'].iloc[0]
        if pd.isna(baseline_time) or baseline_time == 0:
            print(f"Warning: Baseline time is invalid for workload {workload_str}. Skipping speedup plot.")
            continue

        # Process OpenMP runs if available
        has_openmp_data = False
        if not omp_runs.empty:
            omp_runs['speedup'] = baseline_time / omp_runs['mean_exec_time_s']
            omp_runs['efficiency'] = omp_runs['speedup'] / omp_runs['num_threads_or_procs']
            has_openmp_data = True
            
        # Process MPI runs if available
        has_mpi_data = False
        if not mpi_runs.empty:
            mpi_runs['speedup'] = baseline_time / mpi_runs['mean_exec_time_s']
            mpi_runs['efficiency'] = mpi_runs['speedup'] / mpi_runs['num_threads_or_procs']
            has_mpi_data = True
            
        # Skip if neither OpenMP nor MPI data is available
        if not has_openmp_data and not has_mpi_data:
            print(f"Warning: No OpenMP or MPI data for workload {workload_str}. Skipping speedup plot.")
            continue

        # --- Speedup Plot for OpenMP ---
        if has_openmp_data:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=omp_runs, x='num_threads_or_procs', y='speedup', marker='o', label='OpenMP Speedup')
            # Ideal speedup line
            max_threads = omp_runs['num_threads_or_procs'].max()
            if pd.notna(max_threads):
                plt.plot([1, max_threads], [1, max_threads], linestyle='--', color='gray', label='Ideal Speedup')
            
            plt.title(f'OpenMP Speedup vs. Threads\nWorkload: {workload_str}')
            plt.xlabel('Number of Threads')
            plt.ylabel('Speedup (vs Sequential)')
            plt.xticks(omp_runs['num_threads_or_procs'].unique())
            plt.legend()
            plt.tight_layout()
            plot_filename_speedup = os.path.join(plots_dir, f"speedup_openmp_{workload_str}.png")
            plt.savefig(plot_filename_speedup)
            plt.close()
            print(f"Saved OpenMP speedup plot to {plot_filename_speedup}")

            # --- Efficiency Plot for OpenMP---
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=omp_runs, x='num_threads_or_procs', y='efficiency', marker='o', label='OpenMP Efficiency')
            plt.axhline(1.0, linestyle='--', color='gray', label='Ideal Efficiency (100%)')
            plt.title(f'OpenMP Efficiency vs. Threads\nWorkload: {workload_str}')
            plt.xlabel('Number of Threads')
            plt.ylabel('Efficiency')
            plt.ylim(0, 1.1) # Efficiency typically between 0 and 1
            plt.xticks(omp_runs['num_threads_or_procs'].unique())
            plt.legend()
            plt.tight_layout()
            plot_filename_efficiency = os.path.join(plots_dir, f"efficiency_openmp_{workload_str}.png")
            plt.savefig(plot_filename_efficiency)
            plt.close()
            print(f"Saved OpenMP efficiency plot to {plot_filename_efficiency}")
        
        # --- Speedup Plot for MPI ---
        if has_mpi_data:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=mpi_runs, x='num_threads_or_procs', y='speedup', marker='o', label='MPI Speedup')
            # Ideal speedup line
            max_procs = mpi_runs['num_threads_or_procs'].max()
            if pd.notna(max_procs):
                plt.plot([1, max_procs], [1, max_procs], linestyle='--', color='gray', label='Ideal Speedup')
            
            plt.title(f'MPI Speedup vs. Processes\nWorkload: {workload_str}')
            plt.xlabel('Number of Processes')
            plt.ylabel('Speedup (vs Sequential)')
            plt.xticks(mpi_runs['num_threads_or_procs'].unique())
            plt.legend()
            plt.tight_layout()
            plot_filename_speedup = os.path.join(plots_dir, f"speedup_mpi_{workload_str}.png")
            plt.savefig(plot_filename_speedup)
            plt.close()
            print(f"Saved MPI speedup plot to {plot_filename_speedup}")

            # --- Efficiency Plot for MPI---
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=mpi_runs, x='num_threads_or_procs', y='efficiency', marker='o', label='MPI Efficiency')
            plt.axhline(1.0, linestyle='--', color='gray', label='Ideal Efficiency (100%)')
            plt.title(f'MPI Efficiency vs. Processes\nWorkload: {workload_str}')
            plt.xlabel('Number of Processes')
            plt.ylabel('Efficiency')
            plt.ylim(0, 1.1) # Efficiency typically between 0 and 1
            plt.xticks(mpi_runs['num_threads_or_procs'].unique())
            plt.legend()
            plt.tight_layout()
            plot_filename_efficiency = os.path.join(plots_dir, f"efficiency_mpi_{workload_str}.png")
            plt.savefig(plot_filename_efficiency)
            plt.close()
            print(f"Saved MPI efficiency plot to {plot_filename_efficiency}")
            
        # --- Combined Speedup Plot (OpenMP vs MPI) ---
        if has_openmp_data and has_mpi_data:
            plt.figure(figsize=(10, 6))
            
            # Combine data for plotting
            omp_runs_for_plot = omp_runs.copy()
            omp_runs_for_plot['implementation'] = 'OpenMP'
            mpi_runs_for_plot = mpi_runs.copy()
            mpi_runs_for_plot['implementation'] = 'MPI'
            combined_df = pd.concat([omp_runs_for_plot, mpi_runs_for_plot])
            
            # Plot combined data
            sns.lineplot(data=combined_df, x='num_threads_or_procs', y='speedup', hue='implementation', 
                         marker='o', style='implementation')
            
            # Ideal speedup line
            max_cores = max(omp_runs['num_threads_or_procs'].max(), mpi_runs['num_threads_or_procs'].max())
            if pd.notna(max_cores):
                plt.plot([1, max_cores], [1, max_cores], linestyle='--', color='gray', label='Ideal Speedup')
            
            plt.title(f'OpenMP vs MPI Speedup\nWorkload: {workload_str}')
            plt.xlabel('Number of Threads/Processes')
            plt.ylabel('Speedup (vs Sequential)')
            plt.legend()
            plt.tight_layout()
            plot_filename_combined = os.path.join(plots_dir, f"speedup_combined_{workload_str}.png")
            plt.savefig(plot_filename_combined)
            plt.close()
            print(f"Saved combined speedup plot to {plot_filename_combined}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot speedup and efficiency.")
    parser.add_argument("--csv_file", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/results/parsed_data/aggregated_summary.csv", help="Path to aggregated CSV file.")
    parser.add_argument("--plots_dir", default="/home/belal/Desktop/ASU/CSE355/convolution_performance/convolution_performance/plots", help="Directory to save plots.")
    args = parser.parse_args()
    
    plot_speedup_efficiency(args.csv_file, args.plots_dir)