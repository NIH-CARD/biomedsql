import os
import pandas as pd

if __name__ == '__main__':
    for file in os.listdir('results/experiment_results'):
        if file.endswith('metrics.csv'):
            metrics = pd.read_csv(f'results/experiment_results/{file}')
            seconds = metrics['total_time_mean'].iloc[0] * 546
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            print(f"{metrics['model'].iloc[0]}, {metrics['experiment'].iloc[0]}= {int(hours):02d}:{int(minutes):02d}")
