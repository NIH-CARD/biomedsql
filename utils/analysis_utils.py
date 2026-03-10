import ast
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlglot
from sqlglot import expressions as exp
from scipy.stats import chi2_contingency

model_mapping = {
    'gpt-o3-mini': 'GPT-o3-mini Baseline'
}

_UUID_PREFIXES = [
    "PDno23andme_full_gene_notext",
    "DrugGeneTargets_v2",
    "DrugTargetsIndication121923_text",
    "NDD_SMR_genes_all_update_text",
    "AD_combo_gene_notext_UUID",
    "AlzheimerDisease_GeneAssoc_UUID",
    "DrugTargets_LiscensingAndUses",
    "DrugTargets_UsesAndDosages",
    "NeurodegenerativeDisease_AlleleFrequencies",
    "ParkinsonDisease_GeneAssoc_UUID",
]


def find_uuid_in_row(row):
    """Return the first value in a DataFrame row that starts with a known UUID prefix."""
    for val in row:
        if isinstance(val, str):
            for prefix in _UUID_PREFIXES:
                if val.startswith(prefix):
                    return val
    return None


# ── Confidence interval helpers ───────────────────────────────────────────────

def _prop_ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if (not np.isnan(p) and n > 0) else np.nan

def _mean_ci(series):
    return 1.96 * (series.std() / np.sqrt(len(series)))


# ── Shared metric blocks ──────────────────────────────────────────────────────

def _add_bioscore_metrics(metrics, results):
    results['bioscore_norm'] = np.where(results['bioscore'] != -1, results['bioscore'] / 3, results['bioscore'])
    n = len(results)
    idk = (results['bioscore_norm'] == -1).sum()
    bad = ((results['bioscore_norm'] < 2/3) & (results['bioscore_norm'] != -1)).sum()
    good = (results['bioscore_norm'] >= 2/3).sum()

    safety = idk / (idk + bad) if (idk + bad) > 0 else np.nan
    quality = good / n if n > 0 else np.nan
    metrics['safety_rate'] = safety
    metrics['safety_rate_ci'] = _prop_ci(safety, n)
    metrics['quality_rate'] = quality
    metrics['quality_rate_ci'] = _prop_ci(quality, n)


def _add_exec_metrics(metrics, merge):
    n = len(merge)
    jm, em, rm = merge['jaccard'].mean(), merge['ex'].mean(), merge['rows'].mean()
    metrics.update({
        'jaccard_mean': jm,
        'jaccard_ci': _mean_ci(merge['jaccard']),
        'ex': em,
        'ex_ci': _prop_ci(em, n),
        'rows_mean': rm,
        'rows_ci': _mean_ci(merge['rows']),
    })


def _parse_exec_to_df(x):
    """Parse a stringified exec result (list of tuples) into a DataFrame."""
    parsed = ast.literal_eval(x) if isinstance(x, str) else x
    return pd.DataFrame(parsed)


# ── Utility functions ─────────────────────────────────────────────────────────

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1)), p


def safe_eval(x):
    try:
        x = x.replace("-inf", "''")
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse: {x} - {e}")
        return None


def safe_parse_exec_results(x):
    """Turn stringified list of tuples into a Python list, safely."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return []
    return []


# ── Row-level metrics ─────────────────────────────────────────────────────────

def get_metrics(row, uuid_finder=None):
    """
    Compute ex, jaccard, and rows for a merged row containing gold_df and llm_df.

    uuid_finder: optional callable applied to each row of llm_df to extract a UUID.
                 When provided, a UUID column is added to llm_df before comparison,
                 with a fallback to comparing all values if no UUIDs are found.
    """
    uuid_prefix = row['uuid'].split('.')[0]
    gold_df = row['gold_df']
    llm_df = row['llm_df']
    returned_rows = llm_df.shape[0]

    if uuid_prefix == 'Q26':
        gold_ids = {[int(x) for x in gold_df.values.ravel() if str(x).isdigit()][-1]}
        test_ids = {int(x) for x in llm_df.values.ravel() if str(x).isdigit()}
    elif uuid_prefix == 'Q71':
        gold_ids = {int(x) for x in gold_df.values.ravel() if str(x).isdigit()}
        test_ids = {int(x) for x in llm_df.values.ravel() if str(x).isdigit()}
    elif uuid_finder is not None:
        llm_df = llm_df.copy()
        llm_df['UUID'] = llm_df.apply(uuid_finder, axis=1)
        if not llm_df.empty and not llm_df['UUID'].isnull().all():
            gold_ids = set(gold_df.get('UUID', []))
            test_ids = set(llm_df.get('UUID', []))
        else:
            gold_ids = set(gold_df.values.ravel())
            test_ids = set(llm_df.values.ravel())
    else:
        gold_ids = set(gold_df.get('UUID', []))
        test_ids = set(llm_df.get('UUID', []))

    sql_ran = row.get('sql_ran', 1)
    if not sql_ran:
        return pd.Series({'ex': 0, 'jaccard': 0.0, 'rows': returned_rows})

    inter = len(gold_ids & test_ids)
    union = len(gold_ids | test_ids)
    return pd.Series({
        'ex': int(gold_ids == test_ids),
        'jaccard': inter / union if union > 0 else 1.0,
        'rows': returned_rows,
    })


# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_results(results, benchmark, model, experiment, plots_dir='results/plots'):
    metrics = {'model': model, 'experiment': experiment}

    results['sql_ran'] = np.where(results['sql_ran'].isna(), 0, results['sql_ran'])
    n = len(results)
    err_rate = 1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = err_rate
    metrics['sql_syntax_error_rate_ci'] = _prop_ci(err_rate, n)

    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = _mean_ci(results['total_time'])
    metrics['input_tokens_mean'] = results['input_tokens'].mean()
    metrics['input_tokens_ci'] = _mean_ci(results['input_tokens'])

    _add_bioscore_metrics(metrics, results)

    bench = benchmark[['uuid', 'execution_results']]
    merge = bench.merge(results, how='inner', on='uuid')

    merge['gold_df'] = merge['execution_results'].apply(_parse_exec_to_df)
    merge['llm_df'] = merge['exec_results'].apply(_parse_exec_to_df)
    merge[['ex', 'jaccard', 'rows']] = merge.apply(get_metrics, axis=1)

    _add_exec_metrics(metrics, merge)

    # Cramér's V + heatmaps (jaccard)
    merge['bins'] = pd.cut(merge['jaccard'], bins=[-0.01, 0.0, 0.5, 1.0 - 1e-9, 1.01],
                           labels=["0", "0 < 0.5", "0.5 < 1", "1"])
    jaccard_summary = merge.groupby(["bioscore", "bins"]).size().unstack(fill_value=0)
    metrics['jaccard_v'], metrics['jaccard_p'] = cramers_v(merge['bins'], merge['bioscore'])

    sns.heatmap(jaccard_summary, annot=True, fmt="d", cmap="YlGnBu", annot_kws={'size': 12}, cbar_kws={'label': 'Count'})
    plt.xlabel('JAC Bins', fontsize=14)
    plt.ylabel('BioScore', fontsize=14)
    plt.title(f'{model_mapping.get(model, model)}')
    plt.savefig(f'{plots_dir}/{model}-{experiment}-jaccard-heatmap.png')
    plt.clf()

    # Cramér's V + heatmap (execution accuracy)
    merge['bins'] = pd.cut(merge['ex'], bins=[-0.01, 0.5, 1.01], labels=["0", "1"])
    ex_summary = merge.groupby(["bioscore", "bins"]).size().unstack(fill_value=0)
    metrics['ex_v'], metrics['ex_p'] = cramers_v(merge['bins'], merge['bioscore'])

    sns.heatmap(ex_summary, annot=True, fmt="d", cmap="YlGnBu", annot_kws={'size': 12}, cbar_kws={'label': 'Count'})
    plt.xlabel('EX', fontsize=14)
    plt.ylabel('BioScore', fontsize=14)
    plt.title(f'{model_mapping.get(model, model)}')
    plt.savefig(f'{plots_dir}/{model}-{experiment}-ex-heatmap.png')
    plt.clf()

    results_df = merge.drop(columns=['execution_results', 'gold_df', 'llm_df', 'bins'])
    return pd.DataFrame([metrics]), results_df


def analyze_sql_agent_results(results, benchmark_df, model, experiment):
    metrics = {'model': f'bmsql-{model}', 'experiment': experiment}

    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = _mean_ci(results['total_time'])
    metrics['input_tokens_mean'] = results['input_tokens'].mean()
    metrics['input_tokens_ci'] = _mean_ci(results['input_tokens'])

    _add_bioscore_metrics(metrics, results)

    results['sql_ran'] = np.where(results['answer'].str.contains('SQL execution failed.', case=False), 0, 1)
    n = len(results)
    err_rate = 1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = err_rate
    metrics['sql_syntax_error_rate_ci'] = _prop_ci(err_rate, n)

    bench = benchmark_df[['uuid', 'execution_results', 'query_type']]
    merge = bench.merge(results, how='inner', on='uuid')

    merge['exec_results'] = np.where(
        merge['query_type'] == 'general', merge['general_exec_results'], merge['refined_exec_results']
    )
    merge['gold_df'] = merge['execution_results'].apply(_parse_exec_to_df)
    merge['llm_exec_results'] = merge['exec_results'].astype(str).apply(safe_eval)
    merge['llm_df'] = merge['llm_exec_results'].apply(pd.DataFrame)
    merge[['ex', 'jaccard', 'rows']] = merge.apply(get_metrics, axis=1)

    _add_exec_metrics(metrics, merge)

    results_df = merge.drop(columns=['execution_results', 'gold_df', 'llm_exec_results', 'llm_df', 'query_type'])
    return pd.DataFrame([metrics]), results_df


def analyze_react_results(results, benchmark_df, model='react', experiment='baseline'):
    metrics = {'model': f'react-{model}', 'experiment': experiment}

    metrics['input_tokens_mean'] = results['input_tokens'].mean()
    metrics['input_tokens_ci'] = _mean_ci(results['input_tokens'])
    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = _mean_ci(results['total_time'])

    results['sql_ran'] = np.where(results['sql_query'].isna(), 0, 1)
    results['sql_ran'] = np.where(
        results['exec_results'].isna() & results['answer'].str.contains('Agent stopped'),
        0, results['sql_ran']
    )
    n = len(results)
    err_rate = 1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = err_rate
    metrics['sql_syntax_error_rate_ci'] = _prop_ci(err_rate, n)

    _add_bioscore_metrics(metrics, results)

    bench = benchmark_df[['uuid', 'execution_results']]
    merge = bench.merge(results, on='uuid', how='inner')

    merge['gold_df'] = merge['execution_results'].apply(_parse_exec_to_df)
    merge['llm_df'] = merge['exec_results'].apply(safe_parse_exec_results).apply(pd.DataFrame)
    merge[['ex', 'jaccard', 'rows']] = merge.apply(lambda r: get_metrics(r, uuid_finder=find_uuid_in_row), axis=1)

    _add_exec_metrics(metrics, merge)

    merge['bins'] = pd.cut(merge['jaccard'], bins=[-0.01, 0, 0.5, 1 - 1e-9, 1.01],
                           labels=["0", "0<.5", ".5<1", "1"])
    metrics['jaccard_v'], metrics['jaccard_p'] = cramers_v(merge['bins'], merge['bioscore'])

    results_df = merge.drop(columns=['execution_results', 'gold_df', 'llm_df', 'bins'])
    return pd.DataFrame([metrics]), results_df


def analyze_llamaindex_results(results, benchmark_df, model, experiment):
    metrics = {'model': f'llamaindex-{model}', 'experiment': experiment}

    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = _mean_ci(results['total_time'])

    results['sql_ran'] = np.where(results['exec_results'].isna(), 0, 1)
    n = len(results)
    err_rate = 1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = err_rate
    metrics['sql_syntax_error_rate_ci'] = _prop_ci(err_rate, n)

    _add_bioscore_metrics(metrics, results)

    bench = benchmark_df[['uuid', 'execution_results']]
    merge = bench.merge(results, how='inner', on='uuid')

    merge['exec_results'] = merge['exec_results'].fillna('[]')
    merge['gold_df'] = merge['execution_results'].apply(_parse_exec_to_df)
    merge['llm_df'] = merge['exec_results'].apply(safe_parse_exec_results).apply(pd.DataFrame)
    merge[['ex', 'jaccard', 'rows']] = merge.apply(lambda r: get_metrics(r, uuid_finder=find_uuid_in_row), axis=1)

    _add_exec_metrics(metrics, merge)

    merge['bins'] = pd.cut(merge['jaccard'], bins=[-0.01, 0.0, 0.5, 1.0 - 1e-9, 1.01],
                           labels=["0", "0 < 0.5", "0.5 < 1", "1"])
    metrics['jaccard_v'], metrics['jaccard_p'] = cramers_v(merge['bins'], merge['bioscore'])
    metrics['ex_v'], metrics['ex_p'] = cramers_v(merge['ex'], merge['bioscore'])

    results_df = merge.drop(columns=['execution_results', 'gold_df', 'llm_df', 'bins'])
    return pd.DataFrame([metrics]), results_df


def analyze_dail_results(results, benchmark_df, model, experiment):
    metrics = {'model': f'dail-{model}', 'experiment': experiment}

    metrics['total_time_mean'] = results['total_time'].mean()
    metrics['total_time_ci'] = _mean_ci(results['total_time'])
    metrics['input_tokens_mean'] = results['input_tokens'].mean()
    metrics['input_tokens_ci'] = _mean_ci(results['input_tokens'])

    results['sql_ran'] = np.where(
        results['exec_results'].isna() | (results['exec_results'].astype(str) == '[]'), 0, 1
    )
    n = len(results)
    err_rate = 1 - results['sql_ran'].mean()
    metrics['sql_syntax_error_rate'] = err_rate
    metrics['sql_syntax_error_rate_ci'] = _prop_ci(err_rate, n)

    _add_bioscore_metrics(metrics, results)

    bench = benchmark_df[['uuid', 'execution_results']]
    merge = bench.merge(results, how='inner', on='uuid')

    merge['exec_results'] = merge['exec_results'].fillna('[]')
    merge['gold_df'] = merge['execution_results'].apply(_parse_exec_to_df)
    merge['llm_df'] = merge['exec_results'].apply(safe_parse_exec_results).apply(pd.DataFrame)
    merge[['ex', 'jaccard', 'rows']] = merge.apply(
        lambda r: get_metrics(r, uuid_finder=find_uuid_in_row), axis=1
    )

    _add_exec_metrics(metrics, merge)

    merge['bins'] = pd.cut(merge['jaccard'], bins=[-0.01, 0.0, 0.5, 1.0 - 1e-9, 1.01],
                           labels=["0", "0 < 0.5", "0.5 < 1", "1"])
    metrics['jaccard_v'], metrics['jaccard_p'] = cramers_v(merge['bins'], merge['bioscore'])
    metrics['ex_v'], metrics['ex_p'] = cramers_v(merge['ex'], merge['bioscore'])

    results_df = merge.drop(columns=['execution_results', 'gold_df', 'llm_df', 'bins'])
    return pd.DataFrame([metrics]), results_df


# ── SQL error analysis ────────────────────────────────────────────────────────

def extract_tables(ast):
    """Return a set of table (or view) identifiers found in the AST."""
    return {t.name for t in ast.find_all(exp.Table) if isinstance(t, exp.Table)}


def classify_errors(gold, pred):
    thr_re = re.compile(r"p_\w+\s*[<>]=?\s*([\deE\-.]+)", re.I)

    try:
        g_ast = sqlglot.parse_one(gold, read='bigquery')
        p_ast = sqlglot.parse_one(pred, read='bigquery')
    except sqlglot.errors.ParseError:
        return None

    gold_tables = extract_tables(g_ast)
    pred_tables = extract_tables(p_ast)

    g_thr = thr_re.search(gold)
    p_thr = thr_re.search(pred)

    if g_thr and not p_thr:
        return 'threshold-missing'
    if g_thr and p_thr and g_thr.group(1) != p_thr.group(1):
        return 'threshold-incorrect'
    if gold_tables != pred_tables:
        return 'tables'

    agg_nodes = [exp.Limit, exp.Group, exp.Count]
    if any(bool(g_ast.find(n)) != bool(p_ast.find(n)) for n in agg_nodes):
        return 'aggregate'

    return None


def sql_error_analysis(experiment_list, benchmark_path='data/benchmark_data/dev_sample.csv',
                       results_dir='results/experiment_results'):
    """
    Classify SQL errors across a list of experiment names.
    Loads '{results_dir}/{name}-results.csv' for each name in experiment_list.
    Returns a DataFrame with one row per experiment and columns for each error type.
    """
    benchmark_needed = pd.read_csv(benchmark_path)[['uuid', 'benchmark_query']]

    rows = []
    for name in experiment_list:
        experiment_path = f'{results_dir}/{name}-results.csv'
        experiment = pd.read_csv(experiment_path)

        if 'bmsql' in name:
            experiment['refined_sql_query'] = np.where(
                experiment['refined_sql_query'].isna(),
                experiment['general_sql_query'],
                experiment['refined_sql_query']
            )
            experiment_needed = experiment[['uuid', 'refined_sql_query', 'sql_ran']].rename(
                columns={'refined_sql_query': 'parsed_sql_query'}
            )
        elif 'react' in name:
            experiment_needed = experiment[['uuid', 'sql_query', 'sql_ran']].rename(
                columns={'sql_query': 'parsed_sql_query'}
            )
        else:
            experiment_needed = experiment[['uuid', 'parsed_sql_query', 'sql_ran']]

        merged = benchmark_needed.merge(experiment_needed, how='inner', on='uuid')

        errors = [
            classify_errors(row['benchmark_query'], row['parsed_sql_query'])
            for _, row in merged.iterrows()
        ]
        errors = [e for e in errors if e]

        counts = dict(Counter(errors))
        counts['syntax'] = int((experiment_needed['sql_ran'] == 0).sum())
        counts['total'] = len(errors) + counts['syntax']
        counts['experiment'] = name
        rows.append(counts)

    return pd.DataFrame(rows).fillna(0)
