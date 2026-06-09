import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv('config/.env')

from utils.plot_utils import token_histogram_plot, sql_category_distribution_plot, bio_category_distribution_plot, sql_category_radar_plots
from utils.table_utils import baseline_results_table, interaction_results_table, experiment_results_table, compute_results_table, error_analysis_table, bioscore_comparison_table, template_ex_analysis_table
from utils.experiments_utils import substitute_env_placeholders
from utils.analysis_utils import analyze_by_template_partition

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
        full_benchmark = substitute_env_placeholders(full_benchmark_hf['train'].to_pandas())
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
        benchmark = substitute_env_placeholders(benchmark_hf['train'].to_pandas())
        benchmark.to_csv(benchmark_path, index=None)

    token_histogram_plot(full_benchmark)
    sql_category_distribution_plot(full_benchmark)
    bio_category_distribution_plot(full_benchmark)

    results_bmsql_o3_mini = pd.read_csv('results/experiment_results/bmsql-gpt-o3-mini-baseline-results.csv')
    results_o3_mini_combo = pd.read_csv('results/experiment_results/gpt-o3-mini-combo-results.csv')
    results_react_o3_mini = pd.read_csv('results/experiment_results/react-gpt-o3-mini-baseline-results.csv')
    results_o3_mini_baseline = pd.read_csv('results/experiment_results/gpt-o3-mini-baseline-results.csv')
    results_o3_mini_10shot = pd.read_csv('results/experiment_results/gpt-o3-mini-10-shot-results.csv')

    sql_category_radar_plots(
        benchmark, results_bmsql_o3_mini, results_o3_mini_combo, results_react_o3_mini, results_o3_mini_baseline
    )

    baseline_results_table() 
    interaction_results_table() 
    experiment_results_table()
    compute_results_table()
    error_analysis_table()
    bioscore_comparison_table()

    analyze_by_template_partition(results_o3_mini_baseline, full_benchmark, n_groups=5, save_path='results/parition_results.csv')
    template_ex_analysis_table(results_o3_mini_10shot, results_bmsql_o3_mini)

if __name__ == '__main__':
    main()