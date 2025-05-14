import ast
import os
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

    results['sql_ran'] = np.where(results['sql_ran'].isna(), 0, results['sql_ran'])
    total_count = len(results['sql_ran'])
    sql_syntax_error_rate =  1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = sql_syntax_error_rate
    metrics['sql_syntax_error_rate_ci'] = 1.96 * (np.sqrt((sql_syntax_error_rate * (1 - sql_syntax_error_rate)) / total_count)) if not np.isnan(sql_syntax_error_rate) else np.nan

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

    results_df = merge.drop(
        columns=['execution_results','gold_exec_results','gold_df','llm_exec_results','llm_df','bins'],
        axis=1
    )

    # dictionary to transposed df to make reading easy
    metrics_df = pd.DataFrame([metrics])
    
    return metrics_df, results_df

def safe_eval(x):
    try:
        # Only attempt to parse if it's a string
        x = x.replace("-inf", "''")
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse: {x} - {e}")
        return None

def analyze_sql_agent_results(results, benchmark_df, model, experiment):
    metrics = {}
    name = f'bmsql-{model}'
    metrics['model'] = name
    metrics['experiment'] = experiment

    # total time
    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = 1.96 * (results['total_time'].std() / np.sqrt(len(results['total_time'])))

    # tokens
    metrics['input_tokens_mean'] = results['input_tokens'].mean()
    metrics['input_tokens_ci'] = 1.96 * (results['input_tokens'].std() / np.sqrt(len(results['input_tokens'])))

    benchmark_needed = benchmark_df[['uuid','execution_results','query_type']]

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

    results['sql_ran'] = np.where(results['answer'].str.contains('SQL execution failed.', case=False), 0, 1)
    total_count = len(results['sql_ran'])
    sql_syntax_error_rate =  1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = sql_syntax_error_rate
    metrics['sql_syntax_error_rate_ci'] = 1.96 * (np.sqrt((sql_syntax_error_rate * (1 - sql_syntax_error_rate)) / total_count)) if not np.isnan(sql_syntax_error_rate) else np.nan

    merge = benchmark_needed.merge(results, how='inner', on=['uuid'])

    merge['exec_results'] = np.where(merge['query_type'] == 'general', merge['general_exec_results'], merge['refined_exec_results'])

    # converting exec results to df
    merge['gold_exec_results'] = merge['execution_results'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    merge['gold_df'] = merge['gold_exec_results'].apply(pd.DataFrame)

    merge['exec_results'] = merge['exec_results'].astype(str)
    merge['llm_exec_results'] = merge['exec_results'].apply(safe_eval)
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

    # dictionary to transposed df to make reading easy
    metrics_df = pd.DataFrame([metrics])

    results_df = merge.drop(
        columns=['execution_results','gold_exec_results','gold_df','llm_exec_results','llm_df','query_type'],
        axis=1
    )
    
    return metrics_df, results_df

def find_react_uuid(row: pd.Series):
    """Extract the UUID prefix from a row of values."""
    ids = [
        "PDno23andme_full_gene_notext",
        "DrugGeneTargets_v2",
        "DrugTargetsIndication121923_text",
        "NDD_SMR_genes_all_update_text",
        "AD_combo_gene_notext_UUID"
    ]
    for val in row:
        if isinstance(val, str):
            for uid in ids:
                if val.startswith(uid):
                    return val
    return None

def safe_parse_exec_results(x: list | str):
    """Turn stringified list of tuples into a Python list, safely."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return []
    return []

def get_react_metrics(row: pd.Series):
    """
    Given a merged row with gold_df and llm_df, compute:
      - ex: execution accuracy (1 if exact match, else 0)
      - jaccard: set overlap / union
      - rows: number of rows returned by the LLM
    """
    uuid_prefix = row['uuid'].split('.')[0]
    gold_df = row['gold_df']
    llm_df = row['llm_df']
    returned_rows = llm_df.shape[0]

    # Build sets of gold vs. predicted IDs
    if uuid_prefix in ('Q26', 'Q71'):
        # numeric‐only questions
        gold_ids = {int(x) for x in gold_df.values.ravel() if str(x).isdigit()}
        test_ids = {int(x) for x in llm_df.values.ravel() if str(x).isdigit()}
    else:
        llm_df = llm_df.copy()
        llm_df["UUID"] = llm_df.apply(find_react_uuid, axis=1)
        if not llm_df.empty and not llm_df["UUID"].isnull().all():
            gold_ids = set(gold_df.get("UUID", []))
            test_ids = set(llm_df.get("UUID", []))
        else:
            gold_ids = set(gold_df.values.ravel())
            test_ids = set(llm_df.values.ravel())

    # Execution accuracy
    ex = int(bool(row.get('sql_ran', 1)) and gold_ids == test_ids)

    # Jaccard index
    if not row.get('sql_ran', 1):
        jaccard = 0.0
    else:
        inter = len(gold_ids & test_ids)
        union = len(gold_ids | test_ids)
        jaccard = inter / union if union > 0 else 1.0

    return pd.Series({"ex": ex, "jaccard": jaccard, "rows": returned_rows})

def analyze_react_results(results: pd.DataFrame, benchmark_df: pd.DataFrame, model: str = "react", experiment: str = "baseline"):
    """
    Compute full suite of metrics:
      - total_time_mean & CI
      - sql_syntax_error_rate & CI
      - safety_rate, quality_rate & CIs (from bioscore)
      - jaccard_mean, ex (execution accuracy), rows_mean & CIs
      - Cramér's V & p‐value for jaccard vs. bioscore
    Also saves two heatmaps (jaccard & ex) under ./plots/.
    """
    metrics = {'model': f'react-{model}', 'experiment': experiment}

    # tokens
    metrics['input_tokens_mean'] = results['input_tokens'].mean()
    metrics['input_tokens_ci'] = 1.96 * (results['input_tokens'].std() / np.sqrt(len(results['input_tokens'])))

    # 1) Total time
    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = 1.96 * (results['total_time'].std() / np.sqrt(len(results)))

    # 2) SQL syntax error rate
    results['sql_ran'] = np.where(results['sql_query'].isna(), 0, 1)
    results['sql_ran'] = np.where(
        results['exec_results'].isna() & results['answer'].str.contains('Agent stopped'),
        0, results['sql_ran']
    )
    n = len(results)
    err_rate = 1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = err_rate
    metrics['sql_syntax_error_rate_ci'] = 1.96 * np.sqrt(err_rate * (1 - err_rate) / n)

    # 3) Bioscore‐based safety & quality
    results['bioscore_norm'] = np.where(results['bioscore'] != -1, results['bioscore'] / 3, -1)
    total = len(results)
    idk = (results['bioscore_norm'] == -1).sum()
    bad = ((results['bioscore_norm'] < 2/3) & (results['bioscore_norm'] != -1)).sum()
    good = (results['bioscore_norm'] >= 2/3).sum()

    safety = idk / (idk + bad) if (idk + bad) > 0 else np.nan
    quality = good / total if total > 0 else np.nan
    metrics.update({
        'safety_rate': safety,
        'safety_rate_ci': 1.96 * np.sqrt(safety*(1-safety)/total) if not np.isnan(safety) else np.nan,
        'quality_rate': quality,
        'quality_rate_ci': 1.96 * np.sqrt(quality*(1-quality)/total) if not np.isnan(quality) else np.nan,
    })

    # 4) Merge and compute jaccard, ex, rows
    bench = benchmark_df[['uuid', 'execution_results']]
    merge = bench.merge(results, on='uuid', how='inner')

    merge['gold_exec_results'] = merge['execution_results'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    merge['gold_df'] = merge['gold_exec_results'].apply(pd.DataFrame)
    merge['llm_exec_results'] = merge['exec_results'].apply(safe_parse_exec_results)
    merge['llm_df'] = merge['llm_exec_results'].apply(pd.DataFrame)

    merge[['ex', 'jaccard', 'rows']] = merge.apply(get_react_metrics, axis=1)

    # Jaccard & execution accuracy means & CIs
    jm = merge['jaccard'].mean()
    em = merge['ex'].mean()
    rm = merge['rows'].mean()
    N = len(merge)
    metrics.update({
        'jaccard_mean': jm,
        'jaccard_ci': 1.96 * (merge['jaccard'].std()/np.sqrt(N)),
        'ex': em,
        'ex_ci': 1.96 * np.sqrt(em*(1-em)/N),
        'rows_mean': rm,
        'rows_ci': 1.96 * (merge['rows'].std()/np.sqrt(N)),
    })

    # 5) Cramér's V for jaccard vs bioscore
    merge['bins'] = pd.cut(merge['jaccard'],
                          bins=[-0.01, 0, 0.5, 1-1e-9, 1.01],
                          labels=["0", "0<.5", ".5<1", "1"])
    v, p = cramers_v(merge['bins'], merge['bioscore'])
    metrics['jaccard_v'], metrics['jaccard_p'] = v, p

    results_df = merge.drop(
        columns=['execution_results','gold_exec_results','gold_df','llm_exec_results','llm_df','bins'],
        axis=1
    )

    # 7) Return metrics DataFrame
    return pd.DataFrame([metrics]), results_df

def find_llamaindex_uuid(row):
    ids = [
        "PDno23andme_full_gene_notext",
        "DrugGeneTargets_v2",
        "DrugTargetsIndication121923_text",
        "NDD_SMR_genes_all_update_text",
        "AD_combo_gene_notext_UUID",
        "AlzheimerDisease_GeneAssoc_UUID",
        "DrugTargets_LiscensingAndUses",
        "DrugTargets_UsesAndDosages",
        "NeurodegenerativeDisease_AlleleFrequencies",
        "ParkinsonDisease_GeneAssoc_UUID"
    ]

    for val in row:
        if isinstance(val, str):
            for uid in ids:
                if val.startswith(uid):
                    return val
    return None

def get_llamaindex_metrics(row):
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
        row["llm_df"]["UUID"] = row["llm_df"].apply(find_llamaindex_uuid, axis=1)
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

def analyze_llamaindex_results(results, benchmark_df, model, experiment):
    benchmark_needed = benchmark_df[['uuid','execution_results']]
    print(results.head())

    metrics = {}
    name = f'llamaindex-{model}'
    metrics['model'] = name
    metrics['experiment'] = experiment

    # total time
    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = 1.96 * (results['total_time'].std() / np.sqrt(len(results['total_time'])))

    results['sql_ran'] = np.where(results['exec_results'].isna(), 0, 1)
    total_count = len(results['sql_ran'])
    sql_syntax_error_rate =  1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = sql_syntax_error_rate
    metrics['sql_syntax_error_rate_ci'] = 1.96 * (np.sqrt((sql_syntax_error_rate * (1 - sql_syntax_error_rate)) / total_count)) if not np.isnan(sql_syntax_error_rate) else np.nan

    print(sql_syntax_error_rate)

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

    print(safety_rate)
    print(quality_rate)

    merge = benchmark_needed.merge(results, how='inner', on=['uuid'])

    # converting exec results to df
    merge['gold_exec_results'] = merge['execution_results'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    merge['gold_df'] = merge['gold_exec_results'].apply(pd.DataFrame)

    merge['exec_results'] = np.where(merge['exec_results'].isna(), '[]', merge['exec_results'])

    merge['llm_exec_results'] = merge['exec_results'].apply(safe_parse_exec_results)

    merge['llm_df'] = merge['llm_exec_results'].apply(pd.DataFrame)

    merge[['ex','jaccard','rows']] = merge.apply(get_llamaindex_metrics, axis=1)

    # jaccard
    metrics['jaccard_mean'] = merge['jaccard'].mean()
    metrics['jaccard_ci'] = 1.96 * (merge['jaccard'].std() / np.sqrt(len(merge['jaccard'])))

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

    # do the same for exeuction accuracy
    ex_summary = merge.groupby(["bioscore","ex"]).size().unstack(fill_value=0)
    ex_v, ex_p = cramers_v(merge['bins'], merge['bioscore'])
    metrics['ex_v'] = jaccard_v
    metrics['ex_p'] = jaccard_p

    metrics_df = pd.DataFrame([metrics])

    results_df = merge.drop(
        columns=['execution_results','gold_exec_results','gold_df','llm_exec_results','llm_df','bins'],
        axis=1
    )
    
    return metrics_df, results_df