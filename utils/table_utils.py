import os
import pandas as pd

def baseline_results_table(results_dir='results/experiment_results', out_dir='results'):
    model_list = [
        'gpt-4o','gpt-4o-mini','gpt-o3-mini','gemini-2.0-flash','gemini-2.0-flash-lite',
        'claude-3-7-sonnet-20250219','Llama-3.1-70B-Instruct','Meta-Llama-3.1-405B-Instruct',
        'Qwen2.5-Coder-14B-Instruct','Qwen2.5-Coder-32B-Instruct'
    ]

    table = []

    for model in model_list:
        table_row = {}
        if os.path.isfile(f'{results_dir}/{model}-baseline-metrics.csv'):
            model_metrics = pd.read_csv(f'{results_dir}/{model}-baseline-metrics.csv')
            table_row['model'] = model
            table_row['EX'] = str(round(model_metrics['ex'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['ex_ci'].iloc[0]*100, 1)) +')'
            table_row['JAC'] = str(round(model_metrics['jaccard_mean'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['jaccard_ci'].iloc[0]*100, 1)) +')'
            table_row['RQR'] = str(round(model_metrics['quality_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['quality_rate_ci'].iloc[0]*100, 1)) +')'
            table_row['RQR'] = str(round(model_metrics['safety_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['safety_rate_ci'].iloc[0]*100, 1)) +')'
            table_row['SER'] = str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) +')'
            table_row['Tokens'] = str(round(model_metrics['input_tokens_mean'].iloc[0], 0))
            table_row['Time'] = str(round(model_metrics['total_time_mean'].iloc[0], 1)) + '(' + str(round(model_metrics['total_time_ci'].iloc[0], 1)) +')'
            table.append(table_row)
    
    table_df = pd.DataFrame(table)
    table_df.to_csv(f'{out_dir}/baseline_results.csv', index=None)

def interaction_results_table(results_dir='results/experiment_results', out_dir='results'):
    interaction_list = ['react','index','bmsql']

    model_list = ['gpt-4o','gpt-o3-mini','o3-mini','gemini-2.0-flash']

    table = []

    for interaction in interaction_list:
        for model in model_list:
            table_row = {}
            if os.path.isfile(f'{results_dir}/{interaction}-{model}-baseline-metrics.csv'):
                model_metrics = pd.read_csv(f'{results_dir}/{interaction}-{model}-baseline-metrics.csv')
                table_row['model'] = f'{interaction}-{model}'
                table_row['EX'] = str(round(model_metrics['ex'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['ex_ci'].iloc[0]*100, 1)) +')'
                table_row['JAC'] = str(round(model_metrics['jaccard_mean'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['jaccard_ci'].iloc[0]*100, 1)) +')'
                table_row['RQR'] = str(round(model_metrics['quality_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['quality_rate_ci'].iloc[0]*100, 1)) +')'
                table_row['RQR'] = str(round(model_metrics['safety_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['safety_rate_ci'].iloc[0]*100, 1)) +')'
                table_row['SER'] = str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) +')'
                table_row['Tokens'] = str(round(model_metrics['input_tokens_mean'].iloc[0], 0))
                table_row['Time'] = str(round(model_metrics['total_time_mean'].iloc[0], 1)) + '(' + str(round(model_metrics['total_time_ci'].iloc[0], 1)) +')'
                table.append(table_row)
    
    table_df = pd.DataFrame(table)
    table_df.to_csv(f'{out_dir}/interaction_results.csv', index=None)


def experiment_results_table(results_dir='results/experiment_results', model='gpt-o3-mini',out_dir='results'):
    experiment_list = [
        '1-shot','3-shot','5-shot','3-rows','5-rows','threshold','combo'
    ]

    table = []

    for experiment in experiment_list:
        table_row = {}
        if os.path.isfile(f'{results_dir}/{model}-{experiment}-metrics.csv'):
            model_metrics = pd.read_csv(f'{results_dir}/{model}-{experiment}-metrics.csv')
            table_row['model'] = f'{model}-{experiment}'
            table_row['EX'] = str(round(model_metrics['ex'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['ex_ci'].iloc[0]*100, 1)) +')'
            table_row['JAC'] = str(round(model_metrics['jaccard_mean'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['jaccard_ci'].iloc[0]*100, 1)) +')'
            table_row['RQR'] = str(round(model_metrics['quality_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['quality_rate_ci'].iloc[0]*100, 1)) +')'
            table_row['RQR'] = str(round(model_metrics['safety_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['safety_rate_ci'].iloc[0]*100, 1)) +')'
            table_row['SER'] = str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) +')'
            table_row['Tokens'] = str(round(model_metrics['input_tokens_mean'].iloc[0], 0))
            table.append(table_row)
    
    table_df = pd.DataFrame(table)
    table_df.to_csv(f'{out_dir}/{model}_experiment_results.csv', index=None)


def compute_results_table(results_dir='results/experiment_results', interaction='bmsql', model='gpt-o3-mini', out_dir='results'):
    experiment_list = [
        'compute-2','compute-3'
    ]

    table = []

    for experiment in experiment_list:
        table_row = {}
        if os.path.isfile(f'{results_dir}/{interaction}-{model}-{experiment}-metrics.csv'):
            model_metrics = pd.read_csv(f'{results_dir}/{interaction}-{model}-{experiment}-metrics.csv')
            table_row['model'] = f'{model}-{experiment}'
            table_row['EX'] = str(round(model_metrics['ex'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['ex_ci'].iloc[0]*100, 1)) +')'
            table_row['JAC'] = str(round(model_metrics['jaccard_mean'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['jaccard_ci'].iloc[0]*100, 1)) +')'
            table_row['RQR'] = str(round(model_metrics['quality_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['quality_rate_ci'].iloc[0]*100, 1)) +')'
            table_row['RQR'] = str(round(model_metrics['safety_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['safety_rate_ci'].iloc[0]*100, 1)) +')'
            table_row['SER'] = str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) + '(' + str(round(model_metrics['sql_syntax_error_rate'].iloc[0]*100, 1)) +')'
            table_row['Tokens'] = str(round(model_metrics['input_tokens_mean'].iloc[0], 0))
            table.append(table_row)
    
    table_df = pd.DataFrame(table)
    table_df.to_csv(f'{out_dir}/{interaction}-{model}_compute_results.csv', index=None)
