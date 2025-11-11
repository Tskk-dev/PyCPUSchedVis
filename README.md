#  PyCPUSchedVis

A Python tool for visualizing various CPU scheduling algorithms with Gantt charts.

Why? I was bored and I don't like calculating computer scheduling algorithms by hand.

## Features

- **Six CPU Scheduling Algorithms:**
  - FCFS (First Come First Serve)
  - SJF (Shortest Job First - Non-Preemptive)
  - NPP (Non-Preemptive Priority)
  - PP (Preemptive Priority)
  - SRTF (Shortest Remaining Time First - Preemptive SJF)
  - RR (Round Robin)

- **Performance Metrics:**
  - CPU Utilization
  - Waiting Time (per process and average)
  - Turnaround Time (per process and average)

- **Visualizations:**
  - Individual Gantt charts for each algorithm
  - Performance metrics table
  - Comparison charts across all algorithms

## Requirements

```bash
pip install pandas matplotlib numpy
```

## Usage

### Input.py Example

```python
from cpu_scheduler import CPUScheduler

# Define your processes
processes = [
    {'pid': 'P1', 'arrival': 0, 'burst': 5, 'priority': 2},
    {'pid': 'P2', 'arrival': 1, 'burst': 3, 'priority': 1},
    {'pid': 'P3', 'arrival': 2, 'burst': 8, 'priority': 3},
    {'pid': 'P4', 'arrival': 3, 'burst': 6, 'priority': 2},
    {'pid': 'P5', 'arrival': 4, 'burst': 4, 'priority': 1}
]

# Create scheduler and run all algorithms
scheduler = CPUScheduler(processes)
summary = scheduler.run_all_algorithms(quantum=2, output_dir='output')
```

### Process Data Format

Each process should be a dictionary with:
- `pid`: Process ID (string or number)
- `arrival`: Arrival time (integer)
- `burst`: Burst time (integer)
- `priority`: Priority (integer, lower = higher priority) - Optional, required for NPP and PP

### Run Individual Algorithms

```python
scheduler = CPUScheduler(processes)

# FCFS
gantt, waiting_times, turnaround_times = scheduler.fcfs()

# SJF
gantt, waiting_times, turnaround_times = scheduler.sjf()

# Round Robin with quantum=3
gantt, waiting_times, turnaround_times = scheduler.rr(quantum=3)

# Generate Gantt chart for specific algorithm
scheduler.generate_gantt_chart('FCFS', gantt, waiting_times, turnaround_times, 'fcfs_chart.png')
```

## Output

The script generates:
1. Individual Gantt charts for each algorithm (PNG files).
2. A comparison chart showing CPU utilization, waiting time, and turnaround time.

All images are saved in the specified output directory (default: `output/`)


## Customization

### Change Round Robin Quantum

```python
scheduler.run_all_algorithms(quantum=4, output_dir='output')
```

### Custom Output Directory

```python
scheduler.run_all_algorithms(quantum=2, output_dir='my_results')
```


## License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.
