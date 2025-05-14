import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from typeguard import typechecked

@typechecked
def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Compute Cramér's V statistic for association between two categorical variables."""
    confusion = pd.crosstab(x, y)
    chi2, p, _, _ = chi2_contingency(confusion)
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(k - 1, r - 1)), p

@typechecked
def find_uuid(row: pd.Series) -> str | None:
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

@typechecked
def safe_parse_exec_results(x: list | str) -> list:
    """Turn stringified list of tuples into a Python list, safely."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return []
    return []

@typechecked
def get_metrics(row: pd.Series) -> pd.Series:
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
        llm_df["UUID"] = llm_df.apply(find_uuid, axis=1)
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

@typechecked
def score_react_agent(results: pd.DataFrame, benchmark_df: pd.DataFrame, model: str = "react", experiment: str = "baseline"):
    """
    Compute full suite of metrics:
      - total_time_mean & CI
      - sql_syntax_error_rate & CI
      - safety_rate, quality_rate & CIs (from bioscore)
      - jaccard_mean, ex (execution accuracy), rows_mean & CIs
      - Cramér's V & p‐value for jaccard vs. bioscore
    Also saves two heatmaps (jaccard & ex) under ./plots/.
    """
    metrics = {'model': model, 'experiment': experiment}

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

    merge[['ex', 'jaccard', 'rows']] = merge.apply(get_metrics, axis=1)

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

    # 6) Plot heatmaps
    os.makedirs('plots', exist_ok=True)
    js = merge.groupby(['bioscore', 'bins']).size().unstack(fill_value=0)
    sns.heatmap(js, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Jaccard bins')
    plt.ylabel('Bioscore')
    plt.title(f'{model}-{experiment} Jaccard')
    plt.savefig(f'plots/{model}-{experiment}-jaccard.png')
    plt.clf()

    es = merge.groupby(['bioscore', 'ex']).size().unstack(fill_value=0)
    sns.heatmap(es, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel('Execution (ex)')
    plt.ylabel('Bioscore')
    plt.title(f'{model}-{experiment} EX')
    plt.savefig(f'plots/{model}-{experiment}-ex.png')
    plt.clf()

    # 7) Return metrics DataFrame
    return pd.DataFrame([metrics]), merge 