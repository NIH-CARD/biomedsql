import os
import gc
import yaml
import time
import torch
import pandas as pd
from datasets import load_dataset

from handlers.gcp import BQ_CLIENT
from handlers.gcp.big_query import BigQuery

from handlers.llms import AZURE_CLIENT, GEMINI_CLIENT
from handlers.llms.azure_openai_llm import AzureOpenAILLM
from handlers.llms.gemini_llm import GeminiLLM

from handlers.sql.sql_handlers import SQLHandler
from handlers.sql.sql_agent import SQLAgent
from handlers.react.react_sql_agent import ReACTSQLAgent
from handlers.llamaindex.llamaindex_sql import LlamaIndexSQL
from handlers.llamaindex import TABLE_SCHEMAS

from utils.experiments_utils import create_table_info, count_tokens_tiktoken, read_prompts, bioscore_components
from utils.analysis_utils import analyze_sql_agent_results, analyze_react_results, analyze_llamaindex_results

AZURE_OPENAI_MODEL_MAPPING = {
    'gpt-4o': os.environ["AZURE_OPENAI_GPT_4o"],
    'gpt-4o-mini': os.environ["AZURE_OPENAI_GPT_4o_mini"],
    'gpt-o3-mini': os.environ["AZURE_OPENAI_GPT_o3_mini"]
}

def run_pipeline(
        benchmark,
        interaction,
        agent,
        model_name,
        llm_eval_handler,
        llm_eval_model,
        experiment,
        num_passes,
        prompts_dir='prompts',
        out_dir='results',
        plots_dir='results/plots',
        rerun=True
):  
    print(model_name)

    if isinstance(llm_eval_handler, AzureOpenAILLM):
        llm_eval_model_name = AZURE_OPENAI_MODEL_MAPPING.get(llm_eval_model, 'gpt-4o')
    else:
        llm_eval_model_name = llm_eval_model

    sql_gen_prompt, example_query_prompt, nl_answer_prompt, bioscore_prompt = read_prompts(prompt_dir=prompts_dir)

    results_list = []

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/experiment_results', exist_ok=True)
    os.makedirs(f'{plots_dir}/experiment_plots', exist_ok=True)

    if rerun or not os.path.isfile(f'{out_dir}/experiment_results/{interaction}-{model_name}-{experiment}-results.csv'):
        if interaction == 'bmsql':
            columns = [
                'uuid','general_sql_query','general_exec_results','refined_sql_query','refined_exec_results','answer','input_tokens',
                'bioscore','bioscore_input_tokens','bioscore_time','total_time'
            ]
        else:
            columns = [
                'uuid','sql_query','exec_results','answer','bioscore','bioscore_input_tokens','bioscore_time','total_time',
                'input_tokens'
            ]

        for ind, row in benchmark.iterrows():
            row_result = {col: None for col in columns}
            row_result['uuid'] = row['uuid']

            if interaction == 'bmsql':
                total_time_start = time.perf_counter()
                general_sql_query, general_results, refined_sql_query, refined_results, answer, tokens = agent.run_agent(
                    question=row['question'], num_passes=num_passes
                )
                total_time_end = time.perf_counter()

                row_result['general_sql_query'] = general_sql_query
                row_result['general_exec_results'] = general_results
                row_result['refined_sql_query'] = refined_sql_query
                row_result['refined_exec_results'] = refined_results
                row_result['answer'] = answer
                row_result['total_time'] = total_time_end - total_time_start
                row_result['input_tokens'] = tokens

            elif interaction == 'react':
                total_time_start = time.perf_counter()
                resp = agent.run_agent(question=row['question'])
                total_time_end = time.perf_counter()
                exec_recs = resp.results.to_dict(orient="records") if isinstance(resp.results, pd.DataFrame) else []
                answer = resp.explanation

                row_result['sql_query'] = resp.sql_query
                row_result['exec_result'] = exec_recs
                row_result['answer'] = answer
                row_result['input_tokens'] = resp.tokens
                row_result['total_time'] = total_time_end - total_time_start
            
            else:
                total_time_start = time.perf_counter()
                sql_query, exec_results, answer = agent.run_agent(question=row['question'])
                total_time_end = time.perf_counter()

                row_result['sql_query'] = sql_query
                row_result['exec_results'] = exec_results
                row_result['answer'] = answer
                row_result['total_time'] = total_time_end - total_time_start

                schema_context = str()
                for schema in TABLE_SCHEMAS:
                    schema_context += str(schema.context_str)

                full_str = str(row['question']) + str(sql_query) + str(exec_results) + schema_context

                tokens = count_tokens_tiktoken(full_str)
                row_result['input_tokens'] = tokens
            
            question_bioscore_prompt = bioscore_prompt.format(
                question=row['question'],
                gold_ans=row['answer'],
                pred_ans=answer
            )
            bioscore_start = time.perf_counter()
            bioscore, bioscore_input_tokens = bioscore_components(
                llm_eval_handler,
                llm_eval_model_name,
                question_bioscore_prompt
            )
            bioscore_end = time.perf_counter()

            row_result['bioscore'] = float(bioscore)
            row_result['bioscore_input_tokens'] = bioscore_input_tokens
            row_result['bioscore_time'] = bioscore_end - bioscore_start

            results_list.append(row_result)
        results_df = pd.DataFrame(results_list)
    else:
        results_df = pd.read_csv(f'{out_dir}/experiment_results/{interaction}-{model_name}-{experiment}-results.csv')

    if interaction == 'bmsql':
        metrics_df, results = analyze_sql_agent_results(results_df, benchmark, model_name, experiment)
    elif interaction == 'react':
        metrics_df, results = analyze_react_results(results_df, benchmark, model_name, experiment)
    else:
        metrics_df, results = analyze_llamaindex_results(results_df, benchmark, model_name, experiment)
    
    results.to_csv(f'{out_dir}/experiment_results/{interaction}-{model_name}-{experiment}-results.csv', index=False)
    metrics_df.to_csv(f'{out_dir}/experiment_results/{interaction}-{model_name}-{experiment}-metrics.csv', index=False)

def main():
    config_path = 'config/interaction_config.yaml'

    bq_handler = BigQuery(BQ_CLIENT)

    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        paths = config.get("paths", {})
        prompts_dir = paths.get("prompts_dir", "prompts")
        results_dir = paths.get("results_dir", "results")
        benchmark_path = paths.get("benchmark_path", "data/benchmark_data/dev_sample.csv")

        eval_model = config.get("eval_model", {})
        eval_model_provider = eval_model.get("provider", "azure_openai")
        eval_model_name = eval_model.get("model", "gpt-4o")

        experiment_models = config.get("experiment_models")
        experiments = config.get("experiments")

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

        if eval_model_provider == 'azure_openai':
            llm_eval_handler = AzureOpenAILLM(AZURE_CLIENT)
        elif eval_model_provider == 'gemini':
            llm_eval_handler = GeminiLLM(GEMINI_CLIENT)
        else:
            raise ValueError(f'Invalid LLM Provider Passed: {model_provider}')

        if len(experiments) > 0 and len(experiment_models) > 0:
            for model in experiment_models:
                model_interaction = model.get("interaction")
                model_provider = model.get("provider")
                model_name = model.get("model")

                if model_interaction == 'bmsql':
                    table_info_concise = create_table_info(
                        bq_handler=bq_handler,
                        dataset_id='bio_sql_benchmark',
                        num_rows=0
                    )

                    table_info = create_table_info(
                        bq_handler=bq_handler,
                        dataset_id='bio_sql_benchmark',
                        num_rows=5
                    )

                    if model_name == 'gemini-2.0-flash':
                        llm_query_params = {
                            'model': model_name,
                            'max_tokens': 4096,
                            'temperature': 0
                        }
                        llm = GeminiLLM(GEMINI_CLIENT)
                    else:
                        llm_query_params = {
                            'model': AZURE_OPENAI_MODEL_MAPPING.get(model_name, 'gpt-4o'),
                            'max_tokens': 4096,
                            'temperature': 0
                        }
                        llm = AzureOpenAILLM(AZURE_CLIENT)
                    
                    sql_handler = SQLHandler(
                        table_info=table_info,
                        table_info_concise=table_info_concise,
                        llm=llm,
                        llm_query_params=llm_query_params,
                        bq_handler=bq_handler
                    )

                    agent = SQLAgent(
                        sql_handler=sql_handler,
                        max_retries=3
                    )

                elif model_interaction == 'react':
                    agent = ReACTSQLAgent.initialize_agent(
                        model_type=model_provider,
                        model_name=AZURE_OPENAI_MODEL_MAPPING.get(model_name, model_name),
                        credentials_path="config/service_account.json"
                    )

                elif model_interaction == 'index':
                    llamaindex = LlamaIndexSQL.initialize_sql_agent(
                        project_id=os.environ['PROJECT_ID'],
                        database_name=os.environ['DATASET_NAME'],
                        llm_provider='gemini',
                        model_name='gemini-2.0-flash'
                    )

                    agent = LlamaIndexSQL(sql_agent=llamaindex)

                else:
                    raise ValueError(f'Invalid Interaction Passed: {model_interaction}.')

                for experiment in experiments:
                    experiment_name = experiment.get("name")
                    num_passes = experiment.get("num_passes", 1)

                    print(model_interaction, model_provider, model_name, experiment_name)
                    
                    if experiment_name == 'baseline':
                        run_experiment = True
                    elif model_interaction == 'bmsql' and model_name == 'gpt-o3-mini':
                        run_experiment = True
                    else:
                        run_experiment = False
                    
                    if run_experiment:
                        run_pipeline(
                            benchmark=benchmark,
                            interaction=model_interaction,
                            agent=agent,
                            model_name=model_name,
                            llm_eval_handler=llm_eval_handler,
                            llm_eval_model=eval_model_name,
                            experiment=experiment_name,
                            num_passes=num_passes,
                            prompts_dir=prompts_dir,
                            out_dir=results_dir,
                            plots_dir=f'{results_dir}/plots',
                            rerun=True
                        )
        else:
            raise ValueError(f'No Experiment Models or Experiments Passed via config/config.yaml.')
    else:
        raise FileNotFoundError(f'Cannot find file: {config_path}.')
        
if __name__ == '__main__':
    main()