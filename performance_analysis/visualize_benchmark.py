import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# --- Setup ---
PLOT_DIR = "newPlot"
os.makedirs(PLOT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_benchmark_data(scaling_type="strong"):
    """Load and parse the benchmark summary CSV data."""
    csv_file = f"benchmark_data/summary_{scaling_type}.csv"
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows from {csv_file}")
        
        # Clean up data types
        df['threads'] = df['threads'].astype(int)
        numeric_cols = ['avg_time', 'std_time', 'avg_GFLOPS', 'std_GFLOPS', 
                       'avg_bw_MBps', 'std_bw_MBps', 'avg_IPC', 'std_IPC']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Unique image sizes: {sorted(df['image_size'].unique())}")
        print(f"Unique thread counts: {sorted(df['threads'].unique())}")
        print(f"Implementations: {df['implementation'].unique()}")
        
        return df
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        return pd.DataFrame()

def create_scaling_plots(df, scaling_type):
    """Create scaling analysis plots for both strong and weak scaling."""
    print(f"Creating {scaling_type} scaling analysis plots...")
    
    # Colors for implementations
    colors = {
        'serialized': '#1f77b4',  # Blue
        'parallelized': '#ff7f0e',  # Orange
        'mpi_conv': '#2ca02c'  # Green
    }
    
    if scaling_type == "strong":
        # Create figure for speedup plot
        plt.figure(figsize=(12, 8))
        
        # Get serial baseline time (T(1))
        serial_time = df[(df['implementation'] == 'serialized') & 
                        (df['threads'] == 1)]['avg_time'].iloc[0]
        
        # Plot each implementation
        for impl in ['serialized', 'parallelized', 'mpi_conv']:
            impl_data = df[df['implementation'] == impl]
            if len(impl_data) > 0:
                # Sort by threads for proper line connection
                impl_data = impl_data.sort_values('threads')
                
                # Calculate speedup S(p) = T(1) / T(p)
                speedups = serial_time / impl_data['avg_time']
                
                # Calculate speedup standard deviation using error propagation
                # For division: std(x/y) = |x/y| * sqrt((std(x)/x)^2 + (std(y)/y)^2)
                speedup_stds = speedups * np.sqrt(
                    (impl_data['std_time'] / impl_data['avg_time'])**2
                )
                
                # Plot with error bars
                plt.errorbar(
                    impl_data['threads'],
                    speedups,
                    yerr=speedup_stds,
                    marker='o',
                    color=colors[impl],
                    label=impl,
                    linewidth=2,
                    markersize=8,
                    capsize=5,
                    capthick=1,
                    elinewidth=1
                )
        
        # Add ideal speedup line
        max_threads = df['threads'].max()
        plt.plot([1, max_threads], [1, max_threads],
                 'k--', alpha=0.5, label='Ideal Speedup')
        
        # Add serialized baseline (horizontal line at y=1)
        plt.axhline(y=1, color='gray', linestyle=':', label='Serialized Baseline', alpha=0.7)
        
        # Customize plot
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.title('Strong Scaling Speedup Analysis', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set axis limits with some padding
        plt.xlim(0, max_threads * 1.1)
        plt.ylim(0.0, 7)  # Speedup can't exceed thread count in ideal case
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/speedup_vs_threads.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create figure for performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = [
        ('avg_GFLOPS', 'std_GFLOPS', 'GFLOPS'),
        ('avg_bw_MBps', 'std_bw_MBps', 'Bandwidth (MB/s)'),
        ('avg_IPC', 'std_IPC', 'IPC'),
        ('avg_time', 'std_time', 'Execution Time (s)')
    ]
    
    # For weak scaling, create a mapping of threads to image sizes
    if scaling_type == "weak":
        thread_to_size = {
            1: '512x512',  # Add mapping for thread count 1
            2: '512x512',
            4: '1024x1024',
            8: '2048x2048',
            12: '4096x4096',
            16: '8192x8192'
        }
        # Reverse mapping for finding thread count from image size
        size_to_thread = {v: k for k, v in thread_to_size.items()}
    
    for idx, (metric, std_metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # For weak scaling, create a secondary x-axis
        if scaling_type == "weak":
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            
            # Set the secondary x-axis ticks to match thread counts
            thread_counts = sorted(df['threads'].unique())
            ax2.set_xticks(thread_counts)
            # Convert numpy int64 to regular int for dictionary lookup
            ax2.set_xticklabels([thread_to_size[int(t)] for t in thread_counts], rotation=45)
            ax2.set_xlabel('Image Size', fontsize=10, labelpad=10)
        
        for impl in ['serialized', 'parallelized', 'mpi_conv']:
            impl_data = df[df['implementation'] == impl]
            if len(impl_data) > 0:
                impl_data = impl_data.sort_values('threads')
                
                # For weak scaling, handle serialized points differently
                if scaling_type == "weak" and impl == "serialized":
                    # Get unique image sizes
                    unique_sizes = sorted(impl_data['image_size'].unique())
                    # Create a new dataframe with one point per image size
                    spread_data = []
                    for size in unique_sizes:
                        size_data = impl_data[impl_data['image_size'] == size].iloc[0:1].copy()
                        # Set thread count based on image size
                        size_data['threads'] = size_to_thread[size]
                        spread_data.append(size_data)
                    impl_data = pd.concat(spread_data)
                    # Plot without lines
                    ax.errorbar(
                        impl_data['threads'],
                        impl_data[metric],
                        yerr=impl_data[std_metric],
                        marker='o',
                        color=colors[impl],
                        label=impl,
                        linewidth=0,  # No lines
                        markersize=8,
                        capsize=5,
                        capthick=1,
                        elinewidth=1
                    )
                else:
                    ax.errorbar(
                        impl_data['threads'],
                        impl_data[metric],
                        yerr=impl_data[std_metric],
                        marker='o',
                        color=colors[impl],
                        label=impl,
                        linewidth=2,
                        markersize=6,
                        capsize=5
                    )
        
        ax.set_xlabel('Number of Threads', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(f'{scaling_type.capitalize()} Scaling Performance Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{scaling_type}_scaling_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_surface_plots(df):
    """Create 3D surface plots comparing parallel implementations with serial baseline."""
    print("Creating 3D surface plots for weak scaling...")
    
    # Prepare data
    image_sizes = ['512x512', '1024x1024', '2048x2048', '4096x4096', '8192x8192']
    thread_counts = [16, 12, 8, 4, 2]  # Reversed order
    
    # Create meshgrid for plotting (swapped X and Y)
    Y, X = np.meshgrid(range(len(image_sizes)), range(len(thread_counts)))
    
    # Get serial baseline for each image size
    serial_data = df[df['implementation'] == 'serialized']
    Z_serial = np.zeros_like(X, dtype=float)
    for i, size in enumerate(image_sizes):
        data = serial_data[serial_data['image_size'] == size]
        if not data.empty:
            Z_serial[:, i] = data['avg_time'].iloc[0]  # Same time for all thread counts
    
    # Create figure for MPI vs Serial comparison
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(121, projection='3d')
    
    # Get MPI data
    mpi_data = df[df['implementation'] == 'mpi_conv']
    Z_mpi = np.zeros_like(X, dtype=float)
    for i, size in enumerate(image_sizes):
        for j, threads in enumerate(thread_counts):
            data = mpi_data[(mpi_data['image_size'] == size) & (mpi_data['threads'] == threads)]
            if not data.empty:
                Z_mpi[j, i] = data['avg_time'].iloc[0]
    
    # Create smoother surfaces using interpolation
    from scipy.interpolate import griddata
    
    # Create finer mesh for smoother surfaces (swapped X and Y)
    y_fine = np.linspace(0, len(image_sizes)-1, 100)
    x_fine = np.linspace(0, len(thread_counts)-1, 100)
    Y_fine, X_fine = np.meshgrid(y_fine, x_fine)
    
    # Interpolate MPI data
    points = np.column_stack((X.flatten(), Y.flatten()))
    Z_mpi_fine = griddata(points, Z_mpi.flatten(), (X_fine, Y_fine), method='cubic')
    Z_serial_fine = griddata(points, Z_serial.flatten(), (X_fine, Y_fine), method='cubic')
    
    # Plot serial baseline first (so it's behind the MPI surface)
    surf_serial = ax.plot_surface(X_fine, Y_fine, Z_serial_fine, 
                                 color='gray', alpha=0.5, label='Serial',
                                 edgecolor='black', linewidth=0.5)
    # Plot MPI surface
    surf_mpi = ax.plot_surface(X_fine, Y_fine, Z_mpi_fine, 
                              cmap='viridis', alpha=0.8, label='MPI',
                              edgecolor='none')
    
    # Customize plot
    ax.set_xlabel('Thread Count', fontsize=10)
    ax.set_ylabel('Image Size', fontsize=10, labelpad=15)
    ax.set_zlabel('Execution Time (s)', fontsize=10)
    ax.set_title('MPI vs Serial Execution Time', fontsize=12)
    
    # Set custom ticks (swapped X and Y)
    ax.set_xticks(range(len(thread_counts)))
    ax.set_xticklabels(thread_counts)
    ax.set_yticks(range(len(image_sizes)))
    ax.set_yticklabels(image_sizes, rotation=0)
    
    # Create figure for OpenMP vs Serial comparison
    ax = fig.add_subplot(122, projection='3d')
    
    # Get OpenMP data
    omp_data = df[df['implementation'] == 'parallelized']
    Z_omp = np.zeros_like(X, dtype=float)
    for i, size in enumerate(image_sizes):
        for j, threads in enumerate(thread_counts):
            data = omp_data[(omp_data['image_size'] == size) & (omp_data['threads'] == threads)]
            if not data.empty:
                Z_omp[j, i] = data['avg_time'].iloc[0]
    
    # Interpolate OpenMP data
    Z_omp_fine = griddata(points, Z_omp.flatten(), (X_fine, Y_fine), method='cubic')
    
    # Plot serial baseline first
    surf_serial = ax.plot_surface(X_fine, Y_fine, Z_serial_fine, 
                                 color='gray', alpha=0.5, label='Serial',
                                 edgecolor='black', linewidth=0.5)
    # Plot OpenMP surface
    surf_omp = ax.plot_surface(X_fine, Y_fine, Z_omp_fine, 
                              cmap='plasma', alpha=0.8, label='OpenMP',
                              edgecolor='none')
    
    # Customize plot
    ax.set_xlabel('Thread Count', fontsize=10)
    ax.set_ylabel('Image Size', fontsize=10, labelpad=15)
    ax.set_zlabel('Execution Time (s)', fontsize=10)
    ax.set_title('OpenMP+MPI vs Serial Execution Time', fontsize=12)
    
    # Set custom ticks (swapped X and Y)
    ax.set_xticks(range(len(thread_counts)))
    ax.set_xticklabels(thread_counts)
    ax.set_yticks(range(len(image_sizes)))
    ax.set_yticklabels(image_sizes, rotation=0)
    
    # Remove colorbars
    # fig.colorbar(surf_mpi, ax=fig.axes[0], shrink=0.5, aspect=5)
    # fig.colorbar(surf_omp, ax=fig.axes[1], shrink=0.5, aspect=5)
    plt.subplots_adjust(wspace=0.3)  # Increase horizontal space between subplots
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/weak_scaling_3d_surfaces.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_gflops_plot(df):
    """Create GFLOPS vs Threads plot for strong scaling data."""
    print("Creating GFLOPS vs Threads plot...")
    
    # Colors for implementations
    colors = {
        'serialized': '#1f77b4',  # Blue
        'parallelized': '#ff7f0e',  # Orange
        'mpi_conv': '#2ca02c'  # Green
    }
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline GFLOPS
    serial_gflops = df[(df['implementation'] == 'serialized') & 
                      (df['threads'] == 1)]['avg_GFLOPS'].iloc[0]
    
    # Add horizontal line for serial baseline
    plt.axhline(y=serial_gflops, color=colors['serialized'], linestyle=':', 
                label='Serial Baseline', alpha=0.7)
    
    # Plot each implementation
    for impl in ['serialized', 'parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            # Sort by threads for proper line connection
            impl_data = impl_data.sort_values('threads')
            
            # Plot with error bars
            plt.errorbar(
                impl_data['threads'],
                impl_data['avg_GFLOPS'],
                yerr=impl_data['std_GFLOPS'],
                marker='o',
                color=colors[impl],
                label=impl,
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    # Customize plot
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Throughput (GFLOPS/s)', fontsize=12)
    plt.title('Strong Scaling Throughput Analysis', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    max_threads = df['threads'].max()
    plt.xlim(0, max_threads * 1.1)
    
    # Use log scale for y-axis if mpi_conv values are much higher
    mpi_max = df[df['implementation'] == 'mpi_conv']['avg_GFLOPS'].max()
    other_max = df[df['implementation'] != 'mpi_conv']['avg_GFLOPS'].max()
    if mpi_max > other_max * 5:  # If mpi values are 5x higher than others
        plt.yscale('log')
        plt.ylabel('Throughput (GFLOPS/s) - Log Scale', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/gflops_vs_threads.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_hardware_counter_plots(df):
    """Create hardware counter visualization plots."""
    print("Creating hardware counter visualization plots...")
    
    # Colors for implementations
    colors = {
        'serialized': '#1f77b4',  # Blue
        'parallelized': '#ff7f0e',  # Orange
        'mpi_conv': '#2ca02c'  # Green
    }
    
    # 1. IPC vs Threads Plot
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline IPC
    serial_ipc = df[(df['implementation'] == 'serialized') & 
                   (df['threads'] == 1)]['avg_IPC'].iloc[0]
    
    # Add horizontal line for serial baseline
    plt.axhline(y=serial_ipc, color=colors['serialized'], linestyle=':', 
                label='Serial Baseline', alpha=0.7)
    
    for impl in ['serialized', 'parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            impl_data = impl_data.sort_values('threads')
            plt.errorbar(
                impl_data['threads'],
                impl_data['avg_IPC'],
                yerr=impl_data['std_IPC'],
                marker='o',
                color=colors[impl],
                label=impl,
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Instructions per Cycle (IPC)', fontsize=12)
    plt.title('Strong Scaling IPC Analysis', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/ipc_vs_threads.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. L1 Cache Miss Rate vs Threads Plot
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline L1 miss rate
    serial_data = df[(df['implementation'] == 'serialized') & (df['threads'] == 1)]
    serial_l1_miss_rate = serial_data['avg_L1_misses'].iloc[0] / serial_data['avg_instructions'].iloc[0]
    
    # Add horizontal line for serial baseline
    plt.axhline(y=serial_l1_miss_rate, color=colors['serialized'], linestyle=':', 
                label='Serial Baseline', alpha=0.7)
    
    for impl in ['serialized', 'parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            impl_data = impl_data.sort_values('threads')
            # Calculate L1 miss rate (misses per instruction)
            l1_miss_rate = impl_data['avg_L1_misses'] / impl_data['avg_instructions']
            l1_miss_rate_std = l1_miss_rate * np.sqrt(
                (impl_data['std_L1_misses'] / impl_data['avg_L1_misses'])**2 +
                (impl_data['std_instructions'] / impl_data['avg_instructions'])**2
            )
            
            plt.errorbar(
                impl_data['threads'],
                l1_miss_rate,
                yerr=l1_miss_rate_std,
                marker='o',
                color=colors[impl],
                label=impl,
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('L1 Cache Miss Rate (misses/instruction)', fontsize=12)
    plt.title('Strong Scaling L1 Cache Miss Rate Analysis', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/l1_miss_rate_vs_threads.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. LLC Cache Miss Rate vs Threads Plot
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline LLC miss rate
    serial_llc_miss_rate = np.where(
        serial_data['avg_LLC_misses'].iloc[0] > 0,
        serial_data['avg_LLC_misses'].iloc[0] / serial_data['avg_instructions'].iloc[0],
        0
    )
    
    # Add horizontal line for serial baseline
    plt.axhline(y=serial_llc_miss_rate, color=colors['serialized'], linestyle=':', 
                label='Serial Baseline', alpha=0.7)
    
    for impl in ['serialized', 'parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            impl_data = impl_data.sort_values('threads')
            # Calculate LLC miss rate (misses per instruction)
            # Handle potential zero values for LLC misses
            llc_miss_rate = np.where(
                impl_data['avg_LLC_misses'] > 0,
                impl_data['avg_LLC_misses'] / impl_data['avg_instructions'],
                0
            )
            llc_miss_rate_std = np.where(
                impl_data['avg_LLC_misses'] > 0,
                llc_miss_rate * np.sqrt(
                    (impl_data['std_LLC_misses'] / impl_data['avg_LLC_misses'])**2 +
                    (impl_data['std_instructions'] / impl_data['avg_instructions'])**2
                ),
                0
            )
            
            plt.errorbar(
                impl_data['threads'],
                llc_miss_rate,
                yerr=llc_miss_rate_std,
                marker='o',
                color=colors[impl],
                label=impl,
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('LLC Cache Miss Rate (misses/instruction)', fontsize=12)
    plt.title('Strong Scaling LLC Cache Miss Rate Analysis', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/llc_miss_rate_vs_threads.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Memory Bandwidth vs Threads Plot
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline bandwidth
    serial_bw = df[(df['implementation'] == 'serialized') & 
                  (df['threads'] == 1)]['avg_bw_MBps'].iloc[0]
    
    # Add horizontal line for serial baseline
    plt.axhline(y=serial_bw, color=colors['serialized'], linestyle=':', 
                label='Serial Baseline', alpha=0.7)
    
    for impl in ['serialized', 'parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            impl_data = impl_data.sort_values('threads')
            plt.errorbar(
                impl_data['threads'],
                impl_data['avg_bw_MBps'],
                yerr=impl_data['std_bw_MBps'],
                marker='o',
                color=colors[impl],
                label=impl,
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Memory Bandwidth (MB/s)', fontsize=12)
    plt.title('Strong Scaling Memory Bandwidth Analysis', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/mem_bandwidth_vs_threads.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_gflops_3d_surface_plot(df):
    """Create 3D surface plot for GFLOPS/s analysis using weak scaling data."""
    print("Creating 3D surface plot for GFLOPS/s analysis...")
    
    # Prepare data
    image_sizes = ['512x512', '1024x1024', '2048x2048', '4096x4096', '8192x8192']
    thread_counts = [16, 12, 8, 4, 2]  # Reversed order
    
    # Create meshgrid for plotting (swapped X and Y)
    Y, X = np.meshgrid(range(len(image_sizes)), range(len(thread_counts)))
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # Get serial baseline for each image size
    serial_data = df[df['implementation'] == 'serialized']
    Z_serial = np.zeros_like(X, dtype=float)
    for i, size in enumerate(image_sizes):
        data = serial_data[serial_data['image_size'] == size]
        if not data.empty:
            Z_serial[:, i] = data['avg_GFLOPS'].iloc[0]  # Same GFLOPS for all thread counts
    
    # Create subplot for each parallel implementation
    for idx, impl in enumerate(['parallelized', 'mpi_conv']):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        
        # Get implementation data
        impl_data = df[df['implementation'] == impl]
        Z = np.zeros_like(X, dtype=float)
        
        # Fill Z matrix with GFLOPS data
        for i, size in enumerate(image_sizes):
            for j, threads in enumerate(thread_counts):
                data = impl_data[(impl_data['image_size'] == size) & (impl_data['threads'] == threads)]
                if not data.empty:
                    Z[j, i] = data['avg_GFLOPS'].iloc[0]
        
        # Create smoother surfaces using interpolation
        from scipy.interpolate import griddata
        
        # Create finer mesh for smoother surfaces
        y_fine = np.linspace(0, len(image_sizes)-1, 100)
        x_fine = np.linspace(0, len(thread_counts)-1, 100)
        Y_fine, X_fine = np.meshgrid(y_fine, x_fine)
        
        # Interpolate data
        points = np.column_stack((X.flatten(), Y.flatten()))
        Z_fine = griddata(points, Z.flatten(), (X_fine, Y_fine), method='cubic')
        Z_serial_fine = griddata(points, Z_serial.flatten(), (X_fine, Y_fine), method='cubic')
        
        
        # Plot parallel implementation surface
        surf_impl = ax.plot_surface(X_fine, Y_fine, Z_fine, 
                                   cmap='viridis' if impl == 'mpi_conv' else 'plasma',
                                   alpha=0.8, label=impl,
                                   edgecolor='none')
        
        # Plot serial baseline first (so it's behind the parallel surface)
        surf_serial = ax.plot_surface(X_fine, Y_fine, Z_serial_fine, 
                                     color='lightgray', alpha=0.3, label='Serial',
                                     edgecolor='black', linewidth=0.3)
                                     
        # Customize plot
        ax.set_xlabel('Thread Count', fontsize=10)
        ax.set_ylabel('Image Size', fontsize=10, labelpad=15)
        ax.set_zlabel('GFLOPS/s', fontsize=10)
        ax.set_title(f'{impl.capitalize()} vs Serial GFLOPS/s', fontsize=12)
        
        # Set custom ticks
        ax.set_xticks(range(len(thread_counts)))
        ax.set_xticklabels(thread_counts)
        ax.set_yticks(range(len(image_sizes)))
        ax.set_yticklabels(image_sizes, rotation=0)
    
    plt.suptitle('Weak Scaling GFLOPS/s Analysis', fontsize=14)
    plt.subplots_adjust(wspace=0.3)  # Increase horizontal space between subplots
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/gflops_3d_surfaces.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_cache_bandwidth_3d_plots(df):
    """Create 3D surface plots for L1 Cache Miss Rate and Memory Bandwidth analysis."""
    print("Creating 3D surface plots for cache and bandwidth analysis...")
    
    # Prepare data
    image_sizes = ['512x512', '1024x1024', '2048x2048', '4096x4096', '8192x8192']
    thread_counts = [16, 12, 8, 4, 2]  # Reversed order
    
    # Create meshgrid for plotting
    Y, X = np.meshgrid(range(len(image_sizes)), range(len(thread_counts)))
    
    # Get serial baseline for each image size
    serial_data = df[df['implementation'] == 'serialized']
    Z_serial_l1 = np.zeros_like(X, dtype=float)
    Z_serial_bw = np.zeros_like(X, dtype=float)
    
    for i, size in enumerate(image_sizes):
        data = serial_data[serial_data['image_size'] == size]
        if not data.empty:
            # Calculate L1 miss rate for serial
            l1_miss_rate = data['avg_L1_misses'].iloc[0] / data['avg_instructions'].iloc[0]
            Z_serial_l1[:, i] = l1_miss_rate
            # Get bandwidth for serial
            Z_serial_bw[:, i] = data['avg_bw_MBps'].iloc[0]
    
    # Create figures for L1 Cache Miss Rate and Memory Bandwidth
    for plot_type, Z_serial, title, zlabel in [
        ('l1_miss_rate', Z_serial_l1, 'L1 Cache Miss Rate', 'Miss Rate (misses/instruction)'),
        ('bandwidth', Z_serial_bw, 'Memory Bandwidth', 'Bandwidth (MB/s)')
    ]:
        print(f"\nCreating {title} plot...")
        fig = plt.figure(figsize=(18, 12))
        
        # Create subplot for each parallel implementation
        for idx, impl in enumerate(['parallelized', 'mpi_conv']):
            ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
            
            # Get implementation data
            impl_data = df[df['implementation'] == impl]
            Z = np.zeros_like(X, dtype=float)
            
            # Fill Z matrix with data
            for i, size in enumerate(image_sizes):
                for j, threads in enumerate(thread_counts):
                    data = impl_data[(impl_data['image_size'] == size) & (impl_data['threads'] == threads)]
                    if not data.empty:
                        if plot_type == 'l1_miss_rate':
                            # Calculate L1 miss rate
                            Z[j, i] = data['avg_L1_misses'].iloc[0] / data['avg_instructions'].iloc[0]
                        else:
                            # Get bandwidth
                            Z[j, i] = data['avg_bw_MBps'].iloc[0]
                            # Print some values for debugging
                            if size == '4096x4096' and threads == 16:
                                print(f"{impl} {size} {threads} threads:")
                                print(f"  Bandwidth: {Z[j, i]:.2f} MB/s")
                                print(f"  GFLOPS: {data['avg_GFLOPS'].iloc[0]:.2f}")
            
            # Create smoother surfaces using interpolation
            from scipy.interpolate import griddata
            
            # Create finer mesh for smoother surfaces
            y_fine = np.linspace(0, len(image_sizes)-1, 100)
            x_fine = np.linspace(0, len(thread_counts)-1, 100)
            Y_fine, X_fine = np.meshgrid(y_fine, x_fine)
            
            # Interpolate data
            points = np.column_stack((X.flatten(), Y.flatten()))
            Z_fine = griddata(points, Z.flatten(), (X_fine, Y_fine), method='cubic')
            Z_serial_fine = griddata(points, Z_serial.flatten(), (X_fine, Y_fine), method='cubic')
            
            # Plot parallel implementation surface
            surf_impl = ax.plot_surface(X_fine, Y_fine, Z_fine, 
                                      cmap='viridis' if impl == 'mpi_conv' else 'plasma',
                                      alpha=0.8, label=impl,
                                      edgecolor='none')
            
            # Plot serial baseline
            surf_serial = ax.plot_surface(X_fine, Y_fine, Z_serial_fine, 
                                        color='lightgray', alpha=0.3, label='Serial',
                                        edgecolor='black', linewidth=0.3)
            
            # Customize plot
            ax.set_xlabel('Thread Count', fontsize=10)
            ax.set_ylabel('Image Size', fontsize=10, labelpad=15)
            ax.set_zlabel(zlabel, fontsize=10)
            ax.set_title(f'{impl.capitalize()} vs Serial {title}', fontsize=12)
            
            # Set custom ticks
            ax.set_xticks(range(len(thread_counts)))
            ax.set_xticklabels(thread_counts)
            ax.set_yticks(range(len(image_sizes)))
            ax.set_yticklabels(image_sizes, rotation=0)
            
            # Print min/max values for debugging
            print(f"\n{impl} {title} range:")
            print(f"  Min: {np.min(Z):.2f}")
            print(f"  Max: {np.max(Z):.2f}")
        
        plt.suptitle(f'Weak Scaling {title} Analysis', fontsize=14)
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{plot_type}_3d_surfaces.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_efficiency_plot(df):
    """Create Efficiency vs Threads plot for strong scaling data."""
    print("Creating Efficiency vs Threads plot...")
    
    # Colors for implementations
    colors = {
        'serialized': '#1f77b4',  # Blue
        'parallelized': '#ff7f0e',  # Orange
        'mpi_conv': '#2ca02c'  # Green
    }
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline time (T(1))
    serial_time = df[(df['implementation'] == 'serialized') & 
                    (df['threads'] == 1)]['avg_time'].iloc[0]
    
    # Calculate serial baseline efficiency (should be 1.0)
    serial_efficiency = 1.0
    
    # Add horizontal line for serial baseline
    plt.axhline(y=serial_efficiency, color=colors['serialized'], linestyle=':', 
                label='Serial Baseline', alpha=0.7)
    
    # Plot each implementation
    for impl in ['serialized', 'parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            # Sort by threads for proper line connection
            impl_data = impl_data.sort_values('threads')
            
            # Calculate speedup S(p) = T(1) / T(p)
            speedups = serial_time / impl_data['avg_time']
            
            # Calculate efficiency = speedup / threads
            efficiencies = speedups / impl_data['threads']
            
            # Calculate efficiency standard deviation using error propagation
            # For division: std(x/y) = |x/y| * sqrt((std(x)/x)^2 + (std(y)/y)^2)
            efficiency_stds = efficiencies * np.sqrt(
                (impl_data['std_time'] / impl_data['avg_time'])**2
            )
            
            # Plot with error bars
            plt.errorbar(
                impl_data['threads'],
                efficiencies,
                yerr=efficiency_stds,
                marker='o',
                color=colors[impl],
                label=impl,
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    # Add ideal efficiency line (horizontal line at y=1)
    max_threads = df['threads'].max()
    plt.axhline(y=1, color='gray', linestyle=':', label='Ideal Efficiency', alpha=0.7)
    
    # Customize plot
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title('Strong Scaling Efficiency Analysis', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    plt.xlim(0, max_threads * 1.1)
    plt.ylim(0, 1.5)  # Efficiency typically ranges from 0 to 1
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/efficiency_vs_threads.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_amdahls_law_plot(df):
    """Create Amdahl's Law curve for strong scaling data."""
    print("Creating Amdahl's Law curve...")
    
    # Colors for implementations
    colors = {
        'serialized': '#1f77b4',  # Blue
        'parallelized': '#ff7f0e',  # Orange
        'mpi_conv': '#2ca02c'  # Green
    }
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline time (T(1))
    serial_time = df[(df['implementation'] == 'serialized') & 
                    (df['threads'] == 1)]['avg_time'].iloc[0]
    
    # Calculate theoretical Amdahl's Law curves for different parallel fractions
    max_threads = df['threads'].max()
    thread_counts = np.linspace(1, max_threads, 100)
    
    # Plot theoretical curves for different parallel fractions
    parallel_fractions = [0.5, 0.7, 0.9, 0.95, 0.99]
    for p in parallel_fractions:
        # Amdahl's Law: Speedup = 1 / ((1-p) + p/n)
        speedup = 1 / ((1-p) + p/thread_counts)
        plt.plot(thread_counts, speedup, '--', alpha=0.5, 
                label=f'Theoretical (p={p:.2f})')
    
    # Plot actual speedup for each implementation
    for impl in ['serialized', 'parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            # Sort by threads for proper line connection
            impl_data = impl_data.sort_values('threads')
            
            # Calculate speedup S(p) = T(1) / T(p)
            speedups = serial_time / impl_data['avg_time']
            
            # Calculate speedup standard deviation using error propagation
            speedup_stds = speedups * np.sqrt(
                (impl_data['std_time'] / impl_data['avg_time'])**2
            )
            
            # Plot with error bars
            plt.errorbar(
                impl_data['threads'],
                speedups,
                yerr=speedup_stds,
                marker='o',
                color=colors[impl],
                label=f'{impl} (Actual)',
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    # Add ideal speedup line
    plt.plot([1, max_threads], [1, max_threads],
             'k--', alpha=0.5, label='Ideal Speedup')
    
    # Customize plot
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title("Amdahl's Law Analysis - Strong Scaling", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    plt.xlim(0, max_threads * 1.1)
    plt.ylim(0, max_threads * 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/amdahls_law.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_gustafsons_law_plot(df):
    """Create Gustafson's Law curve for weak scaling data."""
    print("Creating Gustafson's Law curve...")
    
    # Colors for implementations
    colors = {
        'serialized': '#1f77b4',  # Blue
        'parallelized': '#ff7f0e',  # Orange
        'mpi_conv': '#2ca02c'  # Green
    }
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get serial baseline time (T(1))
    serial_time = df[(df['implementation'] == 'serialized') & 
                    (df['threads'] == 1)]['avg_time'].iloc[0]
    
    # Create secondary x-axis for image sizes
    ax = plt.gca()
    ax2 = ax.twiny()
    
    # Create mapping of threads to image sizes
    thread_to_size = {
        1: '512x512',  # Add mapping for thread count 1
        2: '512x512',
        4: '1024x1024',
        8: '2048x2048',
        12: '4096x4096',
        16: '8192x8192'
    }
    
    # Calculate theoretical Gustafson's Law curves for different serial fractions
    max_threads = df['threads'].max()
    thread_counts = np.linspace(1, max_threads, 100)
    
    # Plot theoretical curves for different serial fractions
    serial_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    for s in serial_fractions:
        # Gustafson's Law: Speedup = n + (1-n)*s
        speedup = thread_counts + (1-thread_counts)*s
        plt.plot(thread_counts, speedup, '--', alpha=0.5,
                label=f'Theoretical (s={s:.2f})')
    
    # Plot actual speedup for each implementation (excluding serialized)
    for impl in ['parallelized', 'mpi_conv']:
        impl_data = df[df['implementation'] == impl]
        if len(impl_data) > 0:
            # Sort by threads for proper line connection
            impl_data = impl_data.sort_values('threads')
            
            # Calculate speedup S(p) = T(1) / T(p)
            speedups = serial_time / impl_data['avg_time']
            
            # Calculate speedup standard deviation using error propagation
            speedup_stds = speedups * np.sqrt(
                (impl_data['std_time'] / impl_data['avg_time'])**2
            )
            
            # Plot with error bars
            plt.errorbar(
                impl_data['threads'],
                speedups,
                yerr=speedup_stds,
                marker='o',
                color=colors[impl],
                label=f'{impl} (Actual)',
                linewidth=2,
                markersize=8,
                capsize=5,
                capthick=1,
                elinewidth=1
            )
    
    # Add ideal speedup line
    plt.plot([1, max_threads], [1, max_threads],
             'k--', alpha=0.5, label='Ideal Speedup')
    
    # Set up secondary x-axis
    ax2.set_xlim(ax.get_xlim())
    thread_counts = sorted(df['threads'].unique())
    ax2.set_xticks(thread_counts)
    # Convert numpy int64 to regular int for dictionary lookup
    ax2.set_xticklabels([thread_to_size[int(t)] for t in thread_counts], rotation=45)
    ax2.set_xlabel('Image Size', fontsize=12, labelpad=10)
    
    # Customize plot
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    plt.title("Gustafson's Law Analysis - Weak Scaling", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    plt.xlim(0, max_threads * 1.1)
    plt.ylim(0, max_threads * 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/gustafsons_law.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    # Process both strong and weak scaling data
    for scaling_type in ["strong", "weak"]:
        print(f"\nProcessing {scaling_type} scaling data...")
        df = load_benchmark_data(scaling_type)
        
        if df.empty:
            print(f"No {scaling_type} scaling data loaded. Skipping...")
            continue
        
        create_scaling_plots(df, scaling_type)
        
        # Create hardware counter plots for strong scaling data
        if scaling_type == "strong":
            create_hardware_counter_plots(df)
            create_gflops_plot(df)
            create_efficiency_plot(df)
            create_amdahls_law_plot(df)
        
        # Create 3D surface plots only for weak scaling
        if scaling_type == "weak":
            create_3d_surface_plots(df)
            create_gflops_3d_surface_plot(df)
            create_cache_bandwidth_3d_plots(df)
            create_gustafsons_law_plot(df)
    
    print(f"\nAnalysis complete! All plots saved to {PLOT_DIR}/")
    print(f"Generated plots:")
    plot_files = [f for f in os.listdir(PLOT_DIR) if f.endswith('.png')]
    for plot_file in sorted(plot_files):
        print(f"  - {plot_file}")

if __name__ == "__main__":
    main() 