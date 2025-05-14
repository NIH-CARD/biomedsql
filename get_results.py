import os
import pandas as pd
from datasets import load_dataset

from utils.plot_utils import token_histogram_plot, sql_category_distribution_plot, sql_category_radar_plots

def main():
    full_benchmark_path = 'data/benchmark_data/BiomedSQL.csv'
    benchmark_path = 'data/benchmark_data/dev_sample.csv'

    if os.path.isfile(full_benchmark_path):
        full_benchmark = pd.read_csv(full_benchmark_path)
    else:
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/benchmark_data', exist_ok=True)
        full_benchmark_hf = load_dataset(
            "csv",
            data_files='https://huggingface.co/datasets/NIH-CARD/BiomedSQL/resolve/main/benchmark_data/BiomedSQL.csv'
        )
        full_benchmark = full_benchmark_hf['train'].to_pandas()
        full_benchmark.to_csv(full_benchmark_path, index=None)
    
    print(full_benchmark.shape)

    if os.path.isfile(benchmark_path):
        benchmark = pd.read_csv(benchmark_path)
    else:
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/benchmark_data', exist_ok=True)
        benchmark_hf = load_dataset(
            "csv",
            data_files='https://huggingface.co/datasets/NIH-CARD/BiomedSQL/resolve/main/benchmark_data/dev_sample.csv'
        )
        benchmark = benchmark_hf['train'].to_pandas()
        benchmark.to_csv(benchmark_path, index=None)

    token_histogram_plot(full_benchmark)
    sql_category_distribution_plot(full_benchmark)

    results_bmsql_o3_mini = pd.read_csv('results/experiment_results/bmsql-gpt-o3-mini-baseline-results.csv')
    results_o3_mini_combo = pd.read_csv('results/experiment_results/gpt-o3-mini-combo-results.csv')
    results_react_o3_mini = pd.read_csv('results/experiment_results/react-gpt-o3-mini-baseline-results.csv')
    results_o3_mini_baseline = pd.read_csv('results/experiment_results/gpt-o3-mini-baseline-results.csv')

    sql_category_radar_plots(
        benchmark, results_bmsql_o3_mini, results_o3_mini_combo, results_react_o3_mini, results_o3_mini_baseline
    )

if __name__ == '__main__':
    main()