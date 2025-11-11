import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
import numpy as np
import os

class CPUScheduler:
    def __init__(self, processes):
        """
        Initialize CPU Scheduler with process data.
        
        Args:
            processes: List of dictionaries with keys: 'pid', 'arrival', 'burst', 'priority' (optional)
        """
        self.processes = processes
        self.results = {}
        
    def fcfs(self):
        """First Come First Serve Algorithm"""
        processes = sorted(self.processes, key=lambda x: (x['arrival'], x['pid']))
        gantt = []
        current_time = 0
        waiting_times = {}
        turnaround_times = {}
        
        for p in processes:
            if current_time < p['arrival']:
                current_time = p['arrival']
            
            start_time = current_time
            end_time = current_time + p['burst']
            gantt.append({'pid': p['pid'], 'start': start_time, 'end': end_time})
            
            waiting_times[p['pid']] = start_time - p['arrival']
            turnaround_times[p['pid']] = end_time - p['arrival']
            current_time = end_time
        
        return gantt, waiting_times, turnaround_times
    
    def sjf(self):
        """Shortest Job First (Non-Preemptive)"""
        processes = [p.copy() for p in self.processes]
        gantt = []
        current_time = 0
        waiting_times = {}
        turnaround_times = {}
        completed = []
        
        while len(completed) < len(processes):
            available = [p for p in processes if p['arrival'] <= current_time and p['pid'] not in completed]
            
            if not available:
                current_time = min([p['arrival'] for p in processes if p['pid'] not in completed])
                continue
            

            selected = min(available, key=lambda x: (x['burst'], x['pid']))
            
            start_time = current_time
            end_time = current_time + selected['burst']
            gantt.append({'pid': selected['pid'], 'start': start_time, 'end': end_time})
            
            waiting_times[selected['pid']] = start_time - selected['arrival']
            turnaround_times[selected['pid']] = end_time - selected['arrival']
            current_time = end_time
            completed.append(selected['pid'])
        
        return gantt, waiting_times, turnaround_times
    
    def npp(self):
        """Non-Preemptive Priority Scheduling (Fixed)"""
        processes = [p.copy() for p in self.processes]
        # Add index to preserve original order
        for i, p in enumerate(processes):
            p['_index'] = i
            
        gantt = []
        current_time = 0
        waiting_times = {}
        turnaround_times = {}
        completed = set()

        # Continue until all processes are done
        while len(completed) < len(processes):
            # Find all processes that have arrived
            available = [p for p in processes if p['arrival'] <= current_time and p['pid'] not in completed]

          
            if not available:
                next_arrival = min([p['arrival'] for p in processes if p['pid'] not in completed])
                current_time = next_arrival
                continue

            selected = min(available, key=lambda x: (x.get('priority', 99), x['arrival'], x['_index']))

            start_time = current_time
            end_time = start_time + selected['burst']
            gantt.append({'pid': selected['pid'], 'start': start_time, 'end': end_time})

       
            waiting_times[selected['pid']] = start_time - selected['arrival']
            turnaround_times[selected['pid']] = end_time - selected['arrival']

            current_time = end_time
            completed.add(selected['pid'])

        return gantt, waiting_times, turnaround_times
    
    def pp(self):
        """Preemptive Priority"""
        processes = [p.copy() for p in self.processes]
        for p in processes:
            p['remaining'] = p['burst']
        
        gantt = []
        current_time = 0
        completed = []
        max_time = max([p['arrival'] + p['burst'] for p in processes]) + 100
        
        while len(completed) < len(processes) and current_time < max_time:
            available = [p for p in processes if p['arrival'] <= current_time and p['pid'] not in completed]
            
            if not available:
                current_time += 1
                continue
            
 
            selected = min(available, key=lambda x: (x.get('priority', 99), x['pid']))
            
 
            if not gantt or gantt[-1]['pid'] != selected['pid']:
                gantt.append({'pid': selected['pid'], 'start': current_time, 'end': current_time + 1})
            else:
                gantt[-1]['end'] = current_time + 1
            
            selected['remaining'] -= 1
            current_time += 1
            
            if selected['remaining'] == 0:
                completed.append(selected['pid'])
        
        # Calculate metrics
        waiting_times = {}
        turnaround_times = {}
        for p in self.processes:
            p_gantt = [g for g in gantt if g['pid'] == p['pid']]
            completion_time = max([g['end'] for g in p_gantt])
            turnaround_times[p['pid']] = completion_time - p['arrival']
            waiting_times[p['pid']] = turnaround_times[p['pid']] - p['burst']
        
        return gantt, waiting_times, turnaround_times
    
    def srtf(self):
        """Shortest Remaining Time First (Preemptive SJF)"""
        processes = [p.copy() for p in self.processes]
        for p in processes:
            p['remaining'] = p['burst']
        
        gantt = []
        current_time = 0
        completed = []
        max_time = max([p['arrival'] + p['burst'] for p in processes]) + 100
        
        while len(completed) < len(processes) and current_time < max_time:
            available = [p for p in processes if p['arrival'] <= current_time and p['pid'] not in completed]
            
            if not available:
                current_time += 1
                continue
            
        
            selected = min(available, key=lambda x: (x['remaining'], x['pid']))
            if not gantt or gantt[-1]['pid'] != selected['pid']:
                gantt.append({'pid': selected['pid'], 'start': current_time, 'end': current_time + 1})
            else:
                gantt[-1]['end'] = current_time + 1
            
            selected['remaining'] -= 1
            current_time += 1
            
            if selected['remaining'] == 0:
                completed.append(selected['pid'])
        
        # Calculate metrics
        waiting_times = {}
        turnaround_times = {}
        for p in self.processes:
            p_gantt = [g for g in gantt if g['pid'] == p['pid']]
            completion_time = max([g['end'] for g in p_gantt])
            turnaround_times[p['pid']] = completion_time - p['arrival']
            waiting_times[p['pid']] = turnaround_times[p['pid']] - p['burst']
        
        return gantt, waiting_times, turnaround_times
    
    def rr(self, quantum=2):
        """Round Robin Algorithm"""
        processes = [p.copy() for p in self.processes]
        for p in processes:
            p['remaining'] = p['burst']
        
        gantt = []
        current_time = 0
        ready_queue = deque()
        completed = []
        added_to_queue = set()
        
 
        processes.sort(key=lambda x: (x['arrival'], x['pid']))
        
        while len(completed) < len(processes):
            for p in processes:
                if p['arrival'] <= current_time and p['pid'] not in added_to_queue and p['pid'] not in completed:
                    ready_queue.append(p)
                    added_to_queue.add(p['pid'])
            
            if not ready_queue:
                current_time += 1
                continue
            
            current_process = ready_queue.popleft()
            exec_time = min(quantum, current_process['remaining'])
            
            gantt.append({
                'pid': current_process['pid'],
                'start': current_time,
                'end': current_time + exec_time
            })
            
            current_process['remaining'] -= exec_time
            current_time += exec_time
            
         
            for p in processes:
                if p['arrival'] <= current_time and p['pid'] not in added_to_queue and p['pid'] not in completed:
                    ready_queue.append(p)
                    added_to_queue.add(p['pid'])
            
            if current_process['remaining'] > 0:
                ready_queue.append(current_process)
            else:
                completed.append(current_process['pid'])
        

        waiting_times = {}
        turnaround_times = {}
        for p in self.processes:
            p_gantt = [g for g in gantt if g['pid'] == p['pid']]
            completion_time = max([g['end'] for g in p_gantt])
            turnaround_times[p['pid']] = completion_time - p['arrival']
            waiting_times[p['pid']] = turnaround_times[p['pid']] - p['burst']
        
        return gantt, waiting_times, turnaround_times
    
    def calculate_cpu_utilization(self, gantt):
        """Calculate CPU utilization percentage"""
        if not gantt:
            return 0
        
        total_time = max([g['end'] for g in gantt])
        busy_time = sum([g['end'] - g['start'] for g in gantt])
        
        return (busy_time / total_time) * 100 if total_time > 0 else 0
    
    def generate_gantt_chart(self, algorithm_name, gantt, waiting_times, turnaround_times, save_path=None):
        """Generate Gantt chart visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        

        unique_pids = sorted(list(set([g['pid'] for g in gantt])))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pids)))
        color_map = {pid: colors[i] for i, pid in enumerate(unique_pids)}
        

        for entry in gantt:
            ax1.barh(0, entry['end'] - entry['start'], left=entry['start'], 
                    height=0.5, color=color_map[entry['pid']], 
                    edgecolor='black', linewidth=1.5)
            # Add process ID in the middle of the bar
            mid_point = (entry['start'] + entry['end']) / 2
            ax1.text(mid_point, 0, entry['pid'], ha='center', va='center', 
                    fontweight='bold', fontsize=10)
        
        # Customize Gantt chart
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_title(f'{algorithm_name} - Gantt Chart', fontsize=14, fontweight='bold')
        ax1.set_yticks([])
        ax1.grid(axis='x', alpha=0.3)
        
     
        max_time = max([g['end'] for g in gantt])
        ax1.set_xticks(range(0, int(max_time) + 1))
        

        cpu_util = self.calculate_cpu_utilization(gantt)
        avg_waiting = sum(waiting_times.values()) / len(waiting_times)
        avg_turnaround = sum(turnaround_times.values()) / len(turnaround_times)
        

        table_data = []
        for p in sorted(self.processes, key=lambda x: x['pid']):
            pid = p['pid']
            table_data.append([
                pid,
                p['arrival'],
                p['burst'],
                p.get('priority', 'N/A'),
                waiting_times[pid],
                turnaround_times[pid]
            ])
        

        table_data.append([
            'Average', '', '', '',
            f'{avg_waiting:.2f}',
            f'{avg_turnaround:.2f}'
        ])
        

        table = ax2.table(cellText=table_data,
                         colLabels=['Process', 'Arrival', 'Burst', 'Priority', 
                                   'Waiting Time', 'Turnaround Time'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        

        for i in range(len(table_data) + 1):
            for j in range(6):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif i == len(table_data):  # Average row
                    cell.set_facecolor('#FFC107')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax2.axis('off')
        

        fig.text(0.5, 0.02, f'CPU Utilization: {cpu_util:.2f}%', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def run_all_algorithms(self, quantum=2, output_dir='output'):
        """Run all scheduling algorithms and generate charts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        algorithms = {
            'FCFS': lambda: self.fcfs(),
            'SJF': lambda: self.sjf(),
            'NPP': lambda: self.npp(),
            'PP': lambda: self.pp(),
            'SRTF': lambda: self.srtf(),
            'RR': lambda: self.rr(quantum)
        }
        
        summary_data = []
        
        for name, algo_func in algorithms.items():
            print(f"\nRunning {name}...")
            gantt, waiting_times, turnaround_times = algo_func()
            

            save_path = os.path.join(output_dir, f'{name}_gantt_chart.png')
            self.generate_gantt_chart(name, gantt, waiting_times, turnaround_times, save_path)
            

            cpu_util = self.calculate_cpu_utilization(gantt)
            avg_waiting = sum(waiting_times.values()) / len(waiting_times)
            avg_turnaround = sum(turnaround_times.values()) / len(turnaround_times)
            
            summary_data.append({
                'Algorithm': name,
                'CPU Utilization (%)': f'{cpu_util:.2f}',
                'Avg Waiting Time': f'{avg_waiting:.2f}',
                'Avg Turnaround Time': f'{avg_turnaround:.2f}'
            })
        

        self.create_comparison_chart(summary_data, output_dir)
        
        print(f"\nAll charts saved to '{output_dir}' directory!")
        return summary_data
    
    def create_comparison_chart(self, summary_data, output_dir):
        """Create a comparison chart for all algorithms"""
        df = pd.DataFrame(summary_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        algorithms = df['Algorithm'].tolist()
        x_pos = np.arange(len(algorithms))
        
    
        cpu_util = [float(x) for x in df['CPU Utilization (%)'].tolist()]
        axes[0].bar(x_pos, cpu_util, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Algorithm', fontweight='bold')
        axes[0].set_ylabel('CPU Utilization (%)', fontweight='bold')
        axes[0].set_title('CPU Utilization Comparison', fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(algorithms, rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        

        avg_waiting = [float(x) for x in df['Avg Waiting Time'].tolist()]
        axes[1].bar(x_pos, avg_waiting, color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Algorithm', fontweight='bold')
        axes[1].set_ylabel('Avg Waiting Time', fontweight='bold')
        axes[1].set_title('Average Waiting Time Comparison', fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(algorithms, rotation=45)
        axes[1].grid(axis='y', alpha=0.3)

        avg_turnaround = [float(x) for x in df['Avg Turnaround Time'].tolist()]
        axes[2].bar(x_pos, avg_turnaround, color='lightgreen', edgecolor='black')
        axes[2].set_xlabel('Algorithm', fontweight='bold')
        axes[2].set_ylabel('Avg Turnaround Time', fontweight='bold')
        axes[2].set_title('Average Turnaround Time Comparison', fontweight='bold')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(algorithms, rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'algorithm_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


