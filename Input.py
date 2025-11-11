
from cpu_scheduler import CPUScheduler

def Input():

    processes = [
        {'pid': 'P1', 'arrival': 1, 'burst': 4, 'priority': 6},
        {'pid': 'P2', 'arrival': 5, 'burst': 12, 'priority': 2},
        {'pid': 'P3', 'arrival': 5, 'burst': 9, 'priority': 4},
        {'pid': 'P4', 'arrival': 11, 'burst': 2, 'priority': 3},
        {'pid': 'P5', 'arrival': 16, 'burst': 10, 'priority': 1},
        {'pid': 'P6', 'arrival': 20, 'burst': 11, 'priority': 5}
    ]
    
    scheduler = CPUScheduler(processes)
    scheduler.run_all_algorithms(quantum=5, output_dir='output')

if __name__ == "__main__":
    Input()
    print("\n" + "="*70)
    print("Processing completed! Check the 'output' directory for results.")
    print("="*70)
