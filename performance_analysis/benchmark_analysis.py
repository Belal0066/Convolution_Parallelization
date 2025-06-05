import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
RESULTS_DIR = "benchmark_results"
PERF_DIR = "perf_data"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Constants ---
FLOPs_PER_PIXEL = 10
# Updated to include all sizes potentially present, based on CSV sample
PIXELS = {
    "512x512": 512 * 512,
    "1024x1024": 1024 * 1024,
    "2048x2048": 2048 * 2048
}
IMPLS = ["serialized", "parallelized", "mpi_conv"] # Define consistent order for hues

# --- 1. Parse benchmark timing CSV files ---
def load_benchmark_results():
    all_data = []
    for file in os.listdir(RESULTS_DIR):
        if not file.endswith(".csv"): continue
        path = os.path.join(RESULTS_DIR, file)
        # Regex to match various parts, ensure threads is captured as part of the initial group if not explicitly _threads_
        m = re.match(r"([a-zA-Z0-9_]+?)_(\d+)_(\d+x\d+)_.*\.csv", file) # Adjusted regex slightly for robustness
        if not m: 
            # Try alternative common naming: impl_size_threads.csv
            m_alt = re.match(r"([a-zA-Z0-9_]+?)_(\d+x\d+)_(\d+)_.*\.csv", file)
            if not m_alt: continue
            impl, size, threads = m_alt.groups()
        else:
            impl, threads, size = m.groups()
            
        # Normalize implementation names if they have suffixes like _conv
        if impl.endswith("_conv") and "mpi" in impl: # e.g. mpi_conv
             impl = "mpi_conv"
        elif "parallel" in impl: # Catch "parallelized" or similar
             impl = "parallelized"
        elif "serial" in impl: # Catch "serialized" or similar
             impl = "serialized"

        if impl not in IMPLS: # If specific parsing leads to unknown impl, skip or warn
            print(f"Warning: Unrecognized implementation '{impl}' from file '{file}'. Skipping.")
            continue

        df = pd.read_csv(path, header=None, names=["Detected_Threads", "Detected_Size", "Time"])
        df['Implementation'] = impl
        df['Threads'] = int(threads) # Use threads from filename as canonical
        df['Size'] = size # Use size from filename as canonical
        all_data.append(df)
        
    if not all_data:
        print("No benchmark CSV files found or parsed. Exiting.")
        exit()
        
    df_all = pd.concat(all_data, ignore_index=True)
    return df_all

df_time = load_benchmark_results()
df_summary = (
    df_time.groupby(['Implementation', 'Threads', 'Size'])
    .agg(avg_time=('Time', 'mean'), std_time=('Time', 'std'), runs=('Time', 'count'))
    .reset_index()
)
df_summary['Threads'] = df_summary['Threads'].astype(int)
df_summary['Pixels'] = df_summary['Size'].map(PIXELS)
df_summary['FLOPs'] = df_summary['Pixels'] * FLOPs_PER_PIXEL

# Handle cases where FLOPs might be NaN (if Size not in PIXELS map)
df_summary['GFLOPS'] = np.nan
mask_flops_calculable = df_summary['FLOPs'].notna() & (df_summary['avg_time'] > 0)
df_summary.loc[mask_flops_calculable, 'GFLOPS'] = \
    df_summary.loc[mask_flops_calculable, 'FLOPs'] / \
    df_summary.loc[mask_flops_calculable, 'avg_time'] / 1e9

# Baseline for Speedup/Efficiency
def get_baseline_metric(size, metric_col):
    base_row = df_summary[(df_summary['Implementation'] == 'serialized') & 
                          (df_summary['Threads'] == 1) & 
                          (df_summary['Size'] == size)]
    return base_row[metric_col].values[0] if not base_row.empty else np.nan

df_summary['baseline_avg_time'] = df_summary['Size'].apply(lambda s: get_baseline_metric(s, 'avg_time'))
df_summary['baseline_std_time'] = df_summary['Size'].apply(lambda s: get_baseline_metric(s, 'std_time'))

df_summary['Speedup'] = df_summary['baseline_avg_time'] / df_summary['avg_time']
df_summary['Efficiency'] = df_summary['Speedup'] / df_summary['Threads']

# Error Propagation
df_summary['std_GFLOPS'] = np.nan
df_summary.loc[mask_flops_calculable, 'std_GFLOPS'] = \
    df_summary.loc[mask_flops_calculable, 'GFLOPS'] * \
    (df_summary.loc[mask_flops_calculable, 'std_time'] / df_summary.loc[mask_flops_calculable, 'avg_time'])

df_summary['std_Speedup'] = np.nan
df_summary['std_Efficiency'] = np.nan

valid_baseline_mask = df_summary['baseline_avg_time'].notna() & (df_summary['baseline_avg_time'] > 0) & \
                      df_summary['avg_time'].notna() & (df_summary['avg_time'] > 0)

rel_err_baseline_time = (df_summary.loc[valid_baseline_mask, 'baseline_std_time'].fillna(0) / 
                         df_summary.loc[valid_baseline_mask, 'baseline_avg_time'])
rel_err_parallel_time = (df_summary.loc[valid_baseline_mask, 'std_time'].fillna(0) / 
                         df_summary.loc[valid_baseline_mask, 'avg_time'])

df_summary.loc[valid_baseline_mask, 'std_Speedup'] = \
    df_summary.loc[valid_baseline_mask, 'Speedup'] * \
    np.sqrt(rel_err_baseline_time**2 + rel_err_parallel_time**2)

df_summary.loc[valid_baseline_mask, 'std_Efficiency'] = \
    df_summary.loc[valid_baseline_mask, 'std_Speedup'] / \
    df_summary.loc[valid_baseline_mask, 'Threads']


# --- 2. Parse perf files ---
def parse_perf_file(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip().replace(',', '')
            if "cycles" in line and "not" not in line: # Add ' GHz' to avoid 'cycles:u' if present
                metrics['cycles'] = int(line.split()[0])
            elif "instructions" in line: # Add ' insn per cycle' to avoid 'instructions:u'
                metrics['instructions'] = int(line.split()[0])
            elif "cache-misses" in line:
                metrics['cache_misses'] = int(line.split()[0])
                match_rate = re.search(r"#\s+([\d.]+%)", line) # More general match for rate
                if match_rate:
                     metrics['cache_miss_rate'] = float(match_rate.group(1).replace('%',''))
            elif "cache-references" in line:
                metrics['cache_refs'] = int(line.split()[0])
            elif "branches" in line and "branch-misses" not in line: # Avoid double counting
                metrics['branches'] = int(line.split()[0])
            elif "branch-misses" in line:
                metrics['branch_misses'] = int(line.split()[0])
                match_rate = re.search(r"#\s+([\d.]+%)", line) # More general match for rate
                if match_rate:
                    metrics['branch_miss_rate'] = float(match_rate.group(1).replace('%',''))
            elif "seconds time elapsed" in line:
                metrics['time_perf'] = float(line.split()[0]) # Avoid conflict with 'Time' from benchmark
    return metrics

perf_data = []
if os.path.exists(PERF_DIR):
    for file in os.listdir(PERF_DIR):
        if not file.endswith(".perf"): continue
        m = re.match(r"([a-zA-Z0-9_]+?)_(\d+)_(\d+x\d+)_.*\.perf", file)
        if not m: 
            m_alt = re.match(r"([a-zA-Z0-9_]+?)_(\d+x\d+)_(\d+)_.*\.perf", file)
            if not m_alt: continue
            impl, size, threads = m_alt.groups()
        else:
            impl, threads, size = m.groups()

        if impl.endswith("_conv") and "mpi" in impl: impl = "mpi_conv"
        elif "parallel" in impl: impl = "parallelized"
        elif "serial" in impl: impl = "serialized"
        if impl not in IMPLS:
            print(f"Warning: Unrecognized impl '{impl}' from perf file '{file}'. Skipping.")
            continue

        path = os.path.join(PERF_DIR, file)
        metrics = parse_perf_file(path)
        metrics.update({'Implementation': impl, 'Threads': int(threads), 'Size': size})
        perf_data.append(metrics)

df_perf = pd.DataFrame(perf_data) if perf_data else pd.DataFrame()

# --- Merge timing + perf data ---
if not df_perf.empty:
    df_full = pd.merge(df_summary, df_perf, on=['Implementation', 'Threads', 'Size'], how='left')
    df_full['IPC'] = df_full['instructions'] / df_full['cycles']
else: # Create df_full with necessary columns if no perf data, to prevent downstream errors
    df_full = df_summary.copy()
    for col in ['cycles', 'instructions', 'cache_misses', 'cache_miss_rate', 
                'cache_refs', 'branches', 'branch_misses', 'branch_miss_rate', 'time_perf', 'IPC']:
        df_full[col] = np.nan


# --- Save merged data for reproducibility ---
df_full.to_csv("benchmark_analysis.csv", index=False)
print("Saved benchmark_analysis.csv")

sns.set(style="whitegrid")
hue_order_all = IMPLS
hue_order_parallel = [impl for impl in IMPLS if impl != "serialized"]

unique_sizes = df_summary['Size'].unique()

for size_to_plot in unique_sizes:
    if pd.isna(size_to_plot):
        continue
    print(f"Generating plots for size: {size_to_plot}")

    # Filter data for the current size
    df_plot_all_impl_s = df_summary[df_summary['Size'] == size_to_plot]
    df_plot_parallel_impl_s = df_summary[
        (df_summary['Size'] == size_to_plot) &
        (df_summary['Implementation'].isin(hue_order_parallel))
    ]
    df_full_plot_all_impl_s = df_full[df_full['Size'] == size_to_plot]
    df_full_plot_parallel_impl_s = df_full[
        (df_full['Size'] == size_to_plot) &
        (df_full['Implementation'].isin(hue_order_parallel))
    ]

    # Helper for Y-axis adjustment
    def adjust_plot_ylim(df_data, metric_col, std_metric_col, is_ratio=True):
        if not df_data.empty and not df_data[metric_col].isna().all():
            y_data_min = (df_data[metric_col] - df_data[std_metric_col].fillna(0))
            y_data_max = (df_data[metric_col] + df_data[std_metric_col].fillna(0))
            plot_min_y, plot_max_y = y_data_min.min(), y_data_max.max()

            if pd.notna(plot_min_y) and pd.notna(plot_max_y):
                y_range = plot_max_y - plot_min_y
                if y_range < 1e-9: y_range = 0.1 # Avoid division by zero for flat lines

                padding = 0.1 * y_range
                final_min_y = plot_min_y - padding
                if is_ratio: final_min_y = max(0, final_min_y) # Ratios like speedup/efficiency >= 0
                final_max_y = plot_max_y + padding
                
                if final_max_y > final_min_y: plt.ylim(final_min_y, final_max_y)
                elif final_max_y == final_min_y : plt.ylim(final_min_y - 0.05 * abs(final_min_y) if final_min_y != 0 else -0.05, final_max_y + 0.05 * abs(final_max_y) if final_max_y !=0 else 0.05)


    # --- Speedup plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot_all_impl_s, x="Threads", y="Speedup", hue="Implementation", hue_order=hue_order_all, marker='o', errorbar=None)
    for impl_name in df_plot_all_impl_s['Implementation'].unique():
        if impl_name in hue_order_all:
            impl_data = df_plot_all_impl_s[df_plot_all_impl_s['Implementation'] == impl_name]
            if not impl_data.empty:
                color = sns.color_palette()[hue_order_all.index(impl_name) % len(sns.color_palette())]
                plt.errorbar(impl_data['Threads'], impl_data['Speedup'], yerr=impl_data['std_Speedup'],
                             fmt='none', ecolor=color, capsize=3, alpha=0.5, elinewidth=1)
    plt.title(f"Speedup vs Threads ({size_to_plot}) - All Implementations")
    plt.ylabel("Speedup"); plt.xlabel("Threads")
    plt.savefig(f"{PLOT_DIR}/speedup_{size_to_plot}_all.png"); plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot_parallel_impl_s, x="Threads", y="Speedup", hue="Implementation", hue_order=hue_order_parallel, marker='o', errorbar=None)
    for impl_name in df_plot_parallel_impl_s['Implementation'].unique():
         if impl_name in hue_order_all: # Use hue_order_all for consistent color mapping
            impl_data = df_plot_parallel_impl_s[df_plot_parallel_impl_s['Implementation'] == impl_name]
            if not impl_data.empty:
                color = sns.color_palette()[hue_order_all.index(impl_name) % len(sns.color_palette())]
                plt.errorbar(impl_data['Threads'], impl_data['Speedup'], yerr=impl_data['std_Speedup'],
                             fmt='none', ecolor=color, capsize=3, alpha=0.5, elinewidth=1)
    plt.title(f"Speedup vs Threads ({size_to_plot}) - Parallel Only")
    adjust_plot_ylim(df_plot_parallel_impl_s, 'Speedup', 'std_Speedup')
    plt.ylabel("Speedup"); plt.xlabel("Threads")
    plt.savefig(f"{PLOT_DIR}/speedup_{size_to_plot}_parallel_only.png"); plt.close()

    # --- Efficiency plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot_all_impl_s, x="Threads", y="Efficiency", hue="Implementation", hue_order=hue_order_all, marker='o', errorbar=None)
    for impl_name in df_plot_all_impl_s['Implementation'].unique():
        if impl_name in hue_order_all:
            impl_data = df_plot_all_impl_s[df_plot_all_impl_s['Implementation'] == impl_name]
            if not impl_data.empty:
                color = sns.color_palette()[hue_order_all.index(impl_name) % len(sns.color_palette())]
                plt.errorbar(impl_data['Threads'], impl_data['Efficiency'], yerr=impl_data['std_Efficiency'],
                             fmt='none', ecolor=color, capsize=3, alpha=0.5, elinewidth=1)
    plt.title(f"Efficiency vs Threads ({size_to_plot}) - All Implementations")
    plt.ylabel("Efficiency"); plt.xlabel("Threads")
    plt.savefig(f"{PLOT_DIR}/efficiency_{size_to_plot}_all.png"); plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot_parallel_impl_s, x="Threads", y="Efficiency", hue="Implementation", hue_order=hue_order_parallel, marker='o', errorbar=None)
    for impl_name in df_plot_parallel_impl_s['Implementation'].unique():
        if impl_name in hue_order_all:
            impl_data = df_plot_parallel_impl_s[df_plot_parallel_impl_s['Implementation'] == impl_name]
            if not impl_data.empty:
                color = sns.color_palette()[hue_order_all.index(impl_name) % len(sns.color_palette())]
                plt.errorbar(impl_data['Threads'], impl_data['Efficiency'], yerr=impl_data['std_Efficiency'],
                             fmt='none', ecolor=color, capsize=3, alpha=0.5, elinewidth=1)
    plt.title(f"Efficiency vs Threads ({size_to_plot}) - Parallel Only")
    adjust_plot_ylim(df_plot_parallel_impl_s, 'Efficiency', 'std_Efficiency')
    plt.ylabel("Efficiency"); plt.xlabel("Threads")
    plt.savefig(f"{PLOT_DIR}/efficiency_{size_to_plot}_parallel_only.png"); plt.close()
    
    # --- GFLOPS plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot_all_impl_s, x="Threads", y="GFLOPS", hue="Implementation", hue_order=hue_order_all, marker='o', errorbar=None)
    for impl_name in df_plot_all_impl_s['Implementation'].unique():
        if impl_name in hue_order_all:
            impl_data = df_plot_all_impl_s[df_plot_all_impl_s['Implementation'] == impl_name]
            if not impl_data.empty:
                color = sns.color_palette()[hue_order_all.index(impl_name) % len(sns.color_palette())]
                plt.errorbar(impl_data['Threads'], impl_data['GFLOPS'], yerr=impl_data['std_GFLOPS'],
                             fmt='none', ecolor=color, capsize=3, alpha=0.5, elinewidth=1)
    plt.title(f"Throughput (GFLOPS) vs Threads ({size_to_plot}) - All Implementations")
    plt.ylabel("GFLOPS"); plt.xlabel("Threads")
    plt.savefig(f"{PLOT_DIR}/gflops_{size_to_plot}_all.png"); plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot_parallel_impl_s, x="Threads", y="GFLOPS", hue="Implementation", hue_order=hue_order_parallel, marker='o', errorbar=None)
    for impl_name in df_plot_parallel_impl_s['Implementation'].unique():
        if impl_name in hue_order_all:
            impl_data = df_plot_parallel_impl_s[df_plot_parallel_impl_s['Implementation'] == impl_name]
            if not impl_data.empty:
                color = sns.color_palette()[hue_order_all.index(impl_name) % len(sns.color_palette())]
                plt.errorbar(impl_data['Threads'], impl_data['GFLOPS'], yerr=impl_data['std_GFLOPS'],
                             fmt='none', ecolor=color, capsize=3, alpha=0.5, elinewidth=1)
    plt.title(f"Throughput (GFLOPS) vs Threads ({size_to_plot}) - Parallel Only")
    adjust_plot_ylim(df_plot_parallel_impl_s, 'GFLOPS', 'std_GFLOPS')
    plt.ylabel("GFLOPS"); plt.xlabel("Threads")
    plt.savefig(f"{PLOT_DIR}/gflops_{size_to_plot}_parallel_only.png"); plt.close()

    if not df_perf.empty: # Only plot perf-related graphs if perf data was loaded
        # --- Cache Miss Rate ---
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_full_plot_all_impl_s, x="Threads", y="cache_miss_rate", hue="Implementation", hue_order=hue_order_all)
        plt.title(f"Cache Miss Rate (%) vs Threads ({size_to_plot}) - All Implementations")
        plt.ylabel("Cache Miss Rate (%)"); plt.xlabel("Threads")
        plt.savefig(f"{PLOT_DIR}/cache_miss_rate_{size_to_plot}_all.png"); plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_full_plot_parallel_impl_s, x="Threads", y="cache_miss_rate", hue="Implementation", hue_order=hue_order_parallel)
        plt.title(f"Cache Miss Rate (%) vs Threads ({size_to_plot}) - Parallel Only")
        plt.ylabel("Cache Miss Rate (%)"); plt.xlabel("Threads")
        plt.savefig(f"{PLOT_DIR}/cache_miss_rate_{size_to_plot}_parallel_only.png"); plt.close()

        # --- Instructions per Cycle (IPC) ---
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_full_plot_all_impl_s, x="Threads", y="IPC", hue="Implementation", hue_order=hue_order_all)
        plt.title(f"Instructions per Cycle (IPC) vs Threads ({size_to_plot}) - All Implementations")
        plt.ylabel("IPC"); plt.xlabel("Threads")
        plt.savefig(f"{PLOT_DIR}/ipc_{size_to_plot}_all.png"); plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_full_plot_parallel_impl_s, x="Threads", y="IPC", hue="Implementation", hue_order=hue_order_parallel)
        plt.title(f"Instructions per Cycle (IPC) vs Threads ({size_to_plot}) - Parallel Only")
        plt.ylabel("IPC"); plt.xlabel("Threads")
        plt.savefig(f"{PLOT_DIR}/ipc_{size_to_plot}_parallel_only.png"); plt.close()
        
print("All plots generated.")