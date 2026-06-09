import os
import pandas as pd
from scipy.stats import pearsonr
from utils.analysis_utils import sql_error_analysis, template_metrics


def _metric_row(model_metrics, name):
    """Build a single formatted result row from a metrics DataFrame."""
    m = model_metrics.iloc[0]

    def fmt(val, ci=None):
        s = str(round(val * 100, 1))
        return s + '(' + str(round(ci * 100, 1)) + ')' if ci is not None else s

    row = {
        'model': name,
        'EX':  fmt(m['ex'], m['ex_ci']),
        'JAC': fmt(m['jaccard_mean'], m['jaccard_ci']),
        'RQR': fmt(m['quality_rate'], m['quality_rate_ci']),
        'SR':  fmt(m['safety_rate'], m['safety_rate_ci']),
        'SER': fmt(m['sql_syntax_error_rate'], m.get('sql_syntax_error_rate_ci')),
    }
    if 'input_tokens_mean' in m:
        row['Tokens'] = str(round(m['input_tokens_mean'], 0))
    if 'total_time_mean' in m:
        row['Time'] = str(round(m['total_time_mean'], 1)) + '(' + str(round(m['total_time_ci'], 1)) + ')'
    return row


def baseline_results_table(results_dir='results/experiment_results', out_dir='results'):
    model_list = [
        'gpt-4o', 'gpt-o3-mini', 'gpt-5.2', 'gemini-2.0-flash', 'gemini-3-pro-preview',
        'claude-3-7-sonnet-20250219', 'claude-opus-4-5-20251101', 'Llama-3.1-70B-Instruct',
        'Meta-Llama-3.1-405B-Instruct', 'Qwen2.5-Coder-14B-Instruct', 'Qwen2.5-Coder-32B-Instruct'
    ]

    table = []
    for model in model_list:
        path = f'{results_dir}/{model}-baseline-metrics.csv'
        if os.path.isfile(path):
            table.append(_metric_row(pd.read_csv(path), model))

    pd.DataFrame(table).to_csv(f'{out_dir}/baseline_results.csv', index=None)


def interaction_results_table(results_dir='results/experiment_results', out_dir='results'):
    interaction_list = ['react', 'index', 'bmsql']
    model_list = ['gpt-4o', 'gpt-o3-mini', 'o3-mini', 'gemini-2.0-flash']

    table = []
    for interaction in interaction_list:
        for model in model_list:
            path = f'{results_dir}/{interaction}-{model}-baseline-metrics.csv'
            if os.path.isfile(path):
                table.append(_metric_row(pd.read_csv(path), f'{interaction}-{model}'))

    pd.DataFrame(table).to_csv(f'{out_dir}/interaction_results.csv', index=None)


def experiment_results_table(results_dir='results/experiment_results', model='gpt-o3-mini', out_dir='results'):
    experiment_list = ['1-shot', '3-shot', '5-shot', '3-rows', '5-rows', 'threshold', 'combo']

    table = []
    for experiment in experiment_list:
        path = f'{results_dir}/{model}-{experiment}-metrics.csv'
        if os.path.isfile(path):
            table.append(_metric_row(pd.read_csv(path), f'{model}-{experiment}'))

    pd.DataFrame(table).to_csv(f'{out_dir}/{model}_experiment_results.csv', index=None)


def compute_results_table(results_dir='results/experiment_results', interaction='bmsql', model='gpt-o3-mini', out_dir='results'):
    experiment_list = ['compute-2', 'compute-3']

    table = []
    for experiment in experiment_list:
        path = f'{results_dir}/{interaction}-{model}-{experiment}-metrics.csv'
        if os.path.isfile(path):
            table.append(_metric_row(pd.read_csv(path), f'{model}-{experiment}'))

    pd.DataFrame(table).to_csv(f'{out_dir}/{interaction}-{model}_compute_results.csv', index=None)


def error_analysis_table(benchmark_path='data/benchmark_data/dev_sample.csv', out_dir='results'):
    """
    Run SQL error classification across experiments and save a formatted table.
    Columns: experiment | syntax | tables | threshold-missing | threshold-incorrect | aggregate | total
    """
    experiment_list = [
        'gpt-4o-baseline',
        'gpt-o3-mini-baseline',
        'claude-3-7-sonnet-20250219-baseline',
        'gpt-o3-mini-combo',
        'react-gpt-o3-mini-baseline',
        'bmsql-gpt-o3-mini-baseline',
    ]

    df = sql_error_analysis(experiment_list, benchmark_path)

    col_order = ['experiment', 'syntax', 'tables', 'threshold-missing', 'threshold-incorrect', 'aggregate', 'total']
    df = df.reindex(columns=[c for c in col_order if c in df.columns])

    int_cols = [c for c in df.columns if c != 'experiment']
    df[int_cols] = df[int_cols].astype(int)

    df.to_csv(f'{out_dir}/error_analysis.csv', index=False)


def template_ex_analysis_table(df_baseline: pd.DataFrame, df_bmsql: pd.DataFrame,
                               baseline_label: str = 'GPT-o3-mini (10-shot)',
                               bmsql_label: str = 'BMSQL-GPT-o3-mini',
                               out_dir: str = 'results') -> pd.DataFrame:
    """
    Print and save per-template EX variability metrics for two result sets.
    Mirrors the output of template_ex_analysis.py as a table function.
    """
    results = [
        template_metrics(df_baseline, baseline_label),
        template_metrics(df_bmsql, bmsql_label),
    ]

    print("Per-template EX metrics\n")
    for r in results:
        n = r["n_templates"]
        print(f"=== {r['label']} ===")
        print(f"  Mixed templates:               {r['mixed']}/{n}")
        print(f"  Template EX range:             {r['ex_min']:.2f}–{r['ex_max']:.2f}")
        print(f"  Template EX mean ± SD:         {r['ex_mean']:.2f} ± {r['ex_sd']:.2f}")
        print(f"  Mean within-template EX var:   {r['mean_var']:.4f}")
        print()

    rows = []
    for r in results:
        n = r["n_templates"]
        rows.append({
            'Setting': r['label'].split(' (')[0],
            'Mixed templates': f"{r['mixed']}/{n}",
            'Template EX range': f"{r['ex_min']:.2f}–{r['ex_max']:.2f}",
            'Template EX mean ± SD': f"{r['ex_mean']:.2f} ± {r['ex_sd']:.2f}",
            'Mean within-template EX variance': f"{r['mean_var']:.4f}",
        })

    table = pd.DataFrame(rows)
    table.to_csv(f'{out_dir}/template_ex_analysis.csv', index=False)
    return table


def bioscore_comparison_table(path='analyst_graded_bioscores.csv', out_dir='results'):
    """
    Build a comparison table of domain expert vs GPT-4o BioScore distributions
    and compute the Spearman correlation between them.

    Expects a CSV with columns: model_score, analyst_score.
    Returns (table_df, spearman_r, p_value).
    """
    df = pd.read_csv(path)
    df['analyst_score'] = df['analyst_score'].round(1)
    df['model_score'] = df['model_score'].round(1)

    expert_counts = df['analyst_score'].value_counts()
    model_counts = df['model_score'].value_counts()

    all_scores = sorted(set(expert_counts.index) | set(model_counts.index))
    table = pd.DataFrame({
        'BioScore':      all_scores,
        'Domain Expert': [int(expert_counts.get(s, 0)) for s in all_scores],
        'GPT-4o':        [int(model_counts.get(s, 0)) for s in all_scores],
    })

    result = pearsonr(df['model_score'], df['analyst_score'])
    r, p = result.statistic, result.pvalue
    ci = result.confidence_interval(confidence_level=0.95)
    print(f"Pearson r = {r:.4f}, p = {p:.4g}, 95% CI: ({ci.low:.4f}, {ci.high:.4f})")

    table.to_csv(f'{out_dir}/bioscore_comparison.csv', index=False)
