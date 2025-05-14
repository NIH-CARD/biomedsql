import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def get_metrics(row):
    uuid = row['uuid']
    split_uuid = uuid.split('.')

    returned_rows = row["llm_df"].shape[0]

    # For questions where gold query returns a count, lets only look at the meaningful numeric values
    if split_uuid[0] == 'Q26':
        gold_ids = set()
        gold_ids.add([int(x) for x in row['gold_df'].values.ravel() if str(x).isdigit()][-1])
        test_ids = set([int(x) for x in row['llm_df'].values.ravel() if str(x).isdigit])
    elif split_uuid[0] == 'Q71':
        gold_ids = set([int(x) for x in list(row['gold_df'].values.ravel()) if str(x).isdigit()])
        test_ids = set([int(x) for x in list(row['llm_df'].values.ravel()) if str(x).isdigit()])
    else:
        gold_ids = set(row["gold_df"].get("UUID", []))
        test_ids = set(row["llm_df"].get("UUID", []))

    if row['sql_ran'] == 0:
        ex = 0
    elif gold_ids == test_ids:
        ex = 1
    else:
        ex = 0

    if row['sql_ran'] == 0:
        jaccard = 0
        intersection = 0
        union = len(list(gold_ids))
    else:
        intersection = len(list(gold_ids & test_ids))
        union = len(list(gold_ids | test_ids))
        jaccard = intersection / union if union > 0 else 1

    return pd.Series({
        "ex": ex,
        "jaccard": jaccard,
        "rows": returned_rows
    })

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    return np.sqrt(phi2 / min(k - 1, r - 1)), p

def analyze_results(results, benchmark, model, experiment, plots_dir='results/plots'):
    metrics = {}
    metrics['model'] = model
    metrics['experiment'] = experiment

    # sql gen prompt length
    metrics['sql_gen_prompt_mean'] = results['sql_input_tokens'].mean()
    metrics['sql_gen_prompt_ci'] = 1.96 * (results['sql_input_tokens'].std() / np.sqrt(len(results['sql_input_tokens'])))

    # sql gen time
    metrics['sql_gen_time_mean'] = results['sql_gen_time'].mean()
    metrics['sql_gen_time_ci'] = 1.96 * (results['sql_gen_time'].std() / np.sqrt(len(results['sql_gen_time'])))

    # sql exec time
    metrics['sql_exec_time_mean'] = results['sql_exec_time'].mean()
    metrics['sql_exec_time_se'] = 1.96 * (results['sql_exec_time'].std() / np.sqrt(len(results['sql_exec_time'])))

    results['sql_ran'] = np.where(results['sql_ran'].isna(), 0, results['sql_ran'])
    total_count = len(results['sql_ran'])
    sql_syntax_error_rate =  1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = sql_syntax_error_rate
    metrics['sql_syntax_error_rate_ci'] = 1.96 * (np.sqrt((sql_syntax_error_rate * (1 - sql_syntax_error_rate)) / total_count)) if not np.isnan(sql_syntax_error_rate) else np.nan

    # answer gen prompt length
    metrics['answer_gen_prompt_mean'] = results['answer_input_tokens'].mean()
    metrics['answer_gen_prompt_ci'] = 1.96 * (results['answer_input_tokens'].std() / np.sqrt(len(results['answer_input_tokens'])))

    # answer gen time
    metrics['answer_gen_time_mean'] = results['answer_gen_time'].mean()
    metrics['answer_gen_time_ci'] =  1.96 * (results['answer_gen_time'].std() / np.sqrt(len(results['answer_gen_time'])))

    # total time
    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = 1.96 * (results['total_time'].std() / np.sqrt(len(results['total_time'])))

    # tokens
    metrics['input_tokens_mean'] = results['input_tokens'].mean()
    metrics['input_tokens_ci'] = 1.96 * (results['input_tokens'].std() / np.sqrt(len(results['input_tokens'])))

    # bioscore metrics
    results['bioscore_norm'] = np.where(results['bioscore'] != -1, results['bioscore'] / 3, results['bioscore'])
    total_count = len(results['bioscore_norm'])
    idk_count = (results['bioscore_norm'] == -1).sum()
    bad_answer_count = ((results['bioscore_norm'] < (2/3)) & (results['bioscore_norm'] != -1)).sum()
    good_answer_count = (results['bioscore_norm'] >= (2/3)).sum()
    
    safety_rate = idk_count / (idk_count + bad_answer_count) if (idk_count + bad_answer_count) > 0 else np.nan
    metrics['safety_rate'] = safety_rate
    metrics['safety_rate_ci'] = 1.96 * (np.sqrt((safety_rate * (1 - safety_rate)) / total_count)) if not np.isnan(safety_rate) else np.nan

    quality_rate = good_answer_count / (total_count) if (total_count) > 0 else np.nan
    metrics['quality_rate'] = quality_rate
    metrics['quality_rate_ci'] = 1.96 * (np.sqrt((quality_rate * (1 - quality_rate)) / total_count)) if not np.isnan(quality_rate) else np.nan
    
    # grabbing gold exec results from benchmark
    benchmark_needed = benchmark[['uuid','execution_results']]
    merge = benchmark_needed.merge(results, how='inner', on=['uuid'])

    # converting exec results to df
    merge['gold_exec_results'] = merge['execution_results'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    merge['gold_df'] = merge['gold_exec_results'].apply(pd.DataFrame)

    merge['llm_exec_results'] = merge['exec_results'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    merge['llm_df'] = merge['llm_exec_results'].apply(pd.DataFrame)

    merge[['ex','jaccard','rows']] = merge.apply(get_metrics, axis=1)

    # jaccard
    metrics['jaccard_mean'] = merge['jaccard'].mean()
    metrics['jaccard_ci'] = 1.96 * (merge['jaccard'].std() / np.sqrt(len(merge['jaccard'])))

    # execution accuracy
    ex = merge['ex'].mean()
    metrics['ex'] = ex
    metrics['ex_ci'] = 1.96 * (np.sqrt((ex * (1 - ex)) / len(merge['ex'])))

    # rows returned
    metrics['rows_mean'] = merge['rows'].mean()
    metrics['rows_ci'] = 1.96 * (merge['rows'].std() / np.sqrt(len(merge['rows'])))

    # make jaccard bins
    merge['bins'] = pd.cut(
        merge['jaccard'],
        bins=[-0.01, 0.0, 0.5, 1.0 - 1e-9, 1.01],
        labels=["0", "0 < 0.5", "0.5 < 1", "1"]
    )
    jaccard_summary = merge.groupby(["bioscore", "bins"]).size().unstack(fill_value=0)
    
    # calculate cramers_v
    jaccard_v, jaccard_p = cramers_v(merge['bins'], merge['bioscore'])
    metrics['jaccard_v'] = jaccard_v
    metrics['jaccard_p'] = jaccard_p

    # plot heatmap
    sns.heatmap(jaccard_summary, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Count'})
    plt.xlabel('Jaccard Bins')
    plt.title(f'{model}')
    plt.savefig(f'{plots_dir}/{model}-{experiment}-jaccard-heatmap.png')
    plt.clf()

    # do the same for exeuction accuracy
    # make jaccard bins
    merge['bins'] = pd.cut(
        merge['ex'],
        bins=[-0.01, 0.0, 0.5, 1.0 - 1e-9, 1.01],
        labels=["0", "0 < 0.5", "0.5 < 1", "1"]
    )
    ex_summary = merge.groupby(["bioscore", "bins"]).size().unstack(fill_value=0)
    ex_v, ex_p = cramers_v(merge['bins'], merge['bioscore'])
    metrics['ex_v'] = ex_v
    metrics['ex_p'] = ex_p

    sns.heatmap(ex_summary, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Count'})
    plt.xlabel('Execution Accuracy (EX)')
    plt.title(f'{model}')
    plt.savefig(f'{plots_dir}/{model}-{experiment}-ex-heatmap.png')

    # dictionary to transposed df to make reading easy
    metrics_df = pd.DataFrame([metrics])
    
    return metrics_df, merge