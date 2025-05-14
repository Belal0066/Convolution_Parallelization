import re
import pandas as pd

def parse_perf_file(filepath):
    """
    Parses a perf stat output file.
    Returns a dictionary of {metric_name: value}.
    """
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # General pattern for most perf counters
        # Handles numbers with commas, optional units, and comments
        # Example: "1,234,567 cycles # 2.33 GHz" or "12345 instructions"
        # Example: "2.34 msec task-clock # 0.998 CPUs utilized"
        patterns = {
            'task-clock': r"([\d,]+\.?\d*)\s+msec\s+task-clock",  # Capture ms directly
            'cycles': r"([\d,]+)\s+cycles",
            'instructions': r"([\d,]+)\s+instructions",
            'cache-references': r"([\d,]+)\s+cache-references",
            'cache-misses': r"([\d,]+)\s+cache-misses",
            'L1-dcache-load-misses': r"([\d,]+)\s+L1-dcache-load-misses",
            'LLC-load-misses': r"([\d,]+)\s+LLC-load-misses", # Sometimes LLC-loads or just LLC-misses
            'LLC-loads': r"([\d,]+)\s+LLC-loads", 
            'branch-instructions': r"([\d,]+)\s+branch-instructions",
            'branch-misses': r"([\d,]+)\s+branch-misses",
            'time_elapsed': r"([\d,]+\.?\d*)\s+seconds time elapsed" # Fallback for time
        }

        for metric_name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value_str = match.group(1).replace(',', '')
                try:
                    metrics[metric_name] = float(value_str)
                except ValueError:
                    print(f"Warning: Could not convert value '{value_str}' for metric '{metric_name}' in file {filepath}")
                    metrics[metric_name] = None
            else:
                # Check for <not counted> or <not supported>
                not_counted_pattern = r"<not counted>\s+" + re.escape(metric_name.replace('_', '-'))
                not_supported_pattern = r"<not supported>\s+" + re.escape(metric_name.replace('_', '-'))
                if re.search(not_counted_pattern, content) or re.search(not_supported_pattern, content):
                    metrics[metric_name] = None # Or specific string like 'not_counted'
                else:
                    # Fallback for LLC-misses if LLC-load-misses is not found (common naming variation)
                    if metric_name == 'LLC-load-misses' and 'LLC-load-misses' not in metrics:
                        llc_miss_match = re.search(r"([\d,]+)\s+LLC-misses", content)
                        if llc_miss_match:
                             value_str = llc_miss_match.group(1).replace(',', '')
                             metrics['LLC-misses_alt'] = float(value_str) # Use a diff name to avoid overwrite

        # Get kernel type from filepath
        is_mpi = 'mpi_' in filepath.lower()
        
        # For MPI, prefer 'time_elapsed' as it represents wall clock time of the entire MPI job
        # For sequential and OpenMP, task-clock is a better measure of CPU time used
        if is_mpi and 'time_elapsed' in metrics and metrics['time_elapsed'] is not None:
            metrics['exec_time_s'] = metrics['time_elapsed']
            # Store task-clock as a separate metric if available
            if 'task-clock' in metrics and metrics['task-clock'] is not None:
                metrics['total_cpu_time_s'] = metrics['task-clock'] / 1000.0
        elif 'task-clock' in metrics and metrics['task-clock'] is not None:
            metrics['exec_time_s'] = metrics['task-clock'] / 1000.0
        elif 'time_elapsed' in metrics and metrics['time_elapsed'] is not None:
            metrics['exec_time_s'] = metrics['time_elapsed']
        else:
            metrics['exec_time_s'] = None
            print(f"Warning: Could not parse execution time from {filepath}")


    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None
    
    return metrics

if __name__ == '__main__':
    # Example usage: python scripts/lib_perf_parser.py results/raw_perf_data/some_file.perf.txt
    import sys
    if len(sys.argv) > 1:
        parsed_data = parse_perf_file(sys.argv[1])
        if parsed_data:
            for k, v in parsed_data.items():
                print(f"{k}: {v}")
    else:
        print("Provide a perf output file path to test the parser.")