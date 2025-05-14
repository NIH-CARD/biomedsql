import os
import tiktoken
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def count_tokens_tiktoken(row) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    string = row['benchmark_query']
    string = str(string) if string is not None else ""
    num_tokens = len(encoding.encode(string))
    return pd.Series({
        "sql_tokens": num_tokens
    })

def calc_response_quality(results):
    total_count = len(results['bioscore_norm'])
    good_answer_count = (results['bioscore_norm'] >= (2/3)).sum()
    quality_rate = good_answer_count / (total_count) if (total_count) > 0 else np.nan
    return quality_rate

def token_histogram_plot(benchmark):
    benchmark['sql_tokens'] = benchmark.apply(count_tokens_tiktoken, axis=1)

    sns.histplot(
        benchmark['sql_tokens'], 
        bins=20, 
        color='#87CEEB',
        edgecolor='black', 
        linewidth=0.8,
        stat="percent"
    )

    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.5)

    median_value = benchmark['sql_tokens'].median()
    plt.axvline(median_value, color='black', linestyle='--', linewidth=1.5, label=f"Median = {median_value}")
    plt.legend(fontsize=14)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.xlim(35, benchmark['sql_tokens'].max() + 5)
    plt.title('SQL Query Token Count Distribution', fontsize=14)
    plt.xlabel('Number of tokens', fontsize=14)
    plt.ylabel('Frequency (%)', fontsize=14)
    plt.savefig('results/plots/token_hist.png')

def sql_category_distribution_plot(benchmark):
    sql_category_counts = dict()

    for ind, row in benchmark.iterrows():
        sql_categories = row['sql_category'].split(', ')
        
        for sql_cat in sql_categories:
            if sql_cat not in sql_category_counts:
                sql_category_counts[sql_cat] = 1
            else:
                sql_category_counts[sql_cat] += 1

    print(sql_category_counts) 

    labels = list(sql_category_counts.keys())
    sizes = list(sql_category_counts.values())
    explode = [0.1] * len(labels)

    # Plot the pie chart
    plt.figure(figsize=(12, 6))
    plt.pie(
        sizes,
        colors=plt.cm.Pastel1.colors,
        labels=labels,
        autopct='%1.1f%%',
        startangle=0,
        pctdistance=0.85,
        explode=explode,
        labeldistance=1.1,
        textprops={'fontsize': 14}
    )

    centre_circle = plt.Circle((0, 0), 0.75, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Add title in the center
    plt.title("SQL Category Distribution", fontsize=14)
    plt.savefig('results/plots/sql_cat_dist.png')

def sql_category_radar_plots(benchmark, bmsql, combo, react, baseline):
    benchmark_needed = benchmark[['uuid','sql_category']]

    plot_df_bmsql = bmsql.merge(benchmark_needed, how='inner', on='uuid')
    plot_df_combo = combo.merge(benchmark_needed, how='inner', on='uuid')
    plot_df_react = react.merge(benchmark_needed, how='inner', on='uuid')
    plot_df_baseline = baseline.merge(benchmark_needed, how='inner', on='uuid')

    sql_categories = [
        'Select','Order By','Multi-Filter','Threshold','Distinct','Similarity Search','Join','Calculate'
    ]

    sql_stats = {}

    for category in sql_categories:
        plot_df_subset_bmsql = plot_df_bmsql[plot_df_bmsql['sql_category'].str.contains(category, case=False)]
        plot_df_subset_combo = plot_df_combo[plot_df_combo['sql_category'].str.contains(category, case=False)]
        plot_df_subset_react = plot_df_react[plot_df_react['sql_category'].str.contains(category, case=False)]
        plot_df_subset_baseline = plot_df_baseline[plot_df_baseline['sql_category'].str.contains(category, case=False)]

        sql_stats[category] = {}
        sql_stats[category]['BMSQL-GPT-o3-mini-ex'] = plot_df_subset_bmsql['ex'].mean()
        sql_stats[category]['GPT-o3-mini-combo-ex'] = plot_df_subset_combo['ex'].mean()
        sql_stats[category]['ReACT-GPT-o3-mini-ex'] = plot_df_subset_react['ex'].mean()
        sql_stats[category]['GPT-o3-mini-baseline-ex'] = plot_df_subset_baseline['ex'].mean()
        sql_stats[category]['BMSQL-GPT-o3-mini-rqr'] = calc_response_quality(plot_df_subset_bmsql)
        sql_stats[category]['GPT-o3-mini-combo-rqr'] = calc_response_quality(plot_df_subset_combo)
        sql_stats[category]['ReACT-GPT-o3-mini-rqr'] = calc_response_quality(plot_df_subset_react)
        sql_stats[category]['GPT-o3-mini-baseline-rqr'] = calc_response_quality(plot_df_subset_baseline)
    
    labels = list(sql_stats.keys())
    values_ex = [v['BMSQL-GPT-o3-mini-ex'] for v in sql_stats.values()]
    values_ex += values_ex[:1]
    values_ex2 = [v['GPT-o3-mini-combo-ex'] for v in sql_stats.values()]
    values_ex2 += values_ex2[:1]
    values_ex3 = [v['ReACT-GPT-o3-mini-ex'] for v in sql_stats.values()]
    values_ex3 += values_ex3[:1]
    values_ex4 = [v['GPT-o3-mini-baseline-ex'] for v in sql_stats.values()]
    values_ex4 += values_ex4[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    fig.set_size_inches(7, 7)
    ax.plot(angles, values_ex, linewidth=2, color="gray", label=f'BMSQL-GPT-o3-mini')
    ax.fill(angles, values_ex, alpha=0.25, color="gray")
    ax.plot(angles, values_ex2, linewidth=2, color="coral", label=f'GPT-o3-mini-combo')
    ax.fill(angles, values_ex2, alpha=0.25, color="coral")
    ax.plot(angles, values_ex3, linewidth=2, color="orchid", label=f'ReAct-GPT-o3-mini')
    ax.fill(angles, values_ex3, alpha=0.25, color="orchid")
    ax.plot(angles, values_ex4, linewidth=2, color="turquoise", label=f'GPT-o3-mini-baseline')
    ax.fill(angles, values_ex4, alpha=0.25, color="turquoise")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])                       
    for angle, lab in zip(angles[:-1], labels):
        if lab in ['Calculate','Similarity Search','Select','Distinct','Join']:
            radius_for_labels = 1.22
        else:
            radius_for_labels = 1.05
            
        ax.text(angle,
            radius_for_labels,
            lab,
            ha="center", va="center",
            fontsize=16)
    ax.spines["polar"].set_color("lightgray")
    ax.spines["polar"].set_linewidth(1.2)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14)
    ax.set_title("EX by SQL Category", y=1.08, fontsize=16)
    ax.grid(True, color='lightgray')
    plt.tight_layout()
    plt.savefig('results/plots/sql_ex.png')

    plt.clf()
    labels = list(sql_stats.keys())
    values_ex = [v['BMSQL-GPT-o3-mini-rqr'] for v in sql_stats.values()]
    values_ex += values_ex[:1]
    values_ex2 = [v['GPT-o3-mini-combo-rqr'] for v in sql_stats.values()]
    values_ex2 += values_ex2[:1]
    values_ex3 = [v['ReACT-GPT-o3-mini-rqr'] for v in sql_stats.values()]
    values_ex3 += values_ex3[:1]
    values_ex4 = [v['GPT-o3-mini-baseline-rqr'] for v in sql_stats.values()]
    values_ex4 += values_ex4[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    fig.set_size_inches(7, 7)
    ax.plot(angles, values_ex, linewidth=2, color="gray", label=f'BMSQL-GPT-o3-mini')
    ax.fill(angles, values_ex, alpha=0.25, color="gray")
    ax.plot(angles, values_ex2, linewidth=2, color="coral", label=f'GPT-o3-mini-combo')
    ax.fill(angles, values_ex2, alpha=0.25, color="coral")
    ax.plot(angles, values_ex3, linewidth=2, color="orchid", label=f'ReAct-GPT-o3-mini')
    ax.fill(angles, values_ex3, alpha=0.25, color="orchid")
    ax.plot(angles, values_ex4, linewidth=2, color="turquoise", label=f'GPT-o3-mini-baseline')
    ax.fill(angles, values_ex4, alpha=0.25, color="turquoise")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])                       
    for angle, lab in zip(angles[:-1], labels):
        if lab in ['Calculate','Similarity Search','Select','Distinct','Join']:
            radius_for_labels = 1.22
        else:
            radius_for_labels = 1.05
            
        ax.text(angle,
            radius_for_labels,
            lab,
            ha="center", va="center",
            fontsize=16)
    ax.spines["polar"].set_color("lightgray")
    ax.spines["polar"].set_linewidth(1.2)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14)
    ax.set_title("RQR by SQL Category", y=1.08, fontsize=16)
    ax.grid(True, color='lightgray')
    plt.tight_layout()
    plt.savefig('results/plots/sql_rqr.png')


