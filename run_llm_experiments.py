import os
import gc
import yaml
import time
import torch
import pandas as pd
from datasets import load_dataset

from handlers.gcp import BQ_CLIENT
from handlers.gcp.big_query import BigQuery

from handlers.llms import AZURE_CLIENT, AZURE_AI_CLIENT, GEMINI_CLIENT, ANTHROPIC_CLIENT
from handlers.llms.hf_llm import HuggingFaceLLM
from handlers.llms.azure_openai_llm import AzureOpenAILLM
from handlers.llms.azure_ai_llm import AzureAILLM
from handlers.llms.gemini_llm import GeminiLLM
from handlers.llms.anthropic_llm import AnthropicLLM
from utils.experiments_utils import (
    create_table_info, read_prompts, generate_sql, parse_sql_query,
    run_sql, generate_answer, bioscore_components, get_examples, get_thresholds
)
from utils.analysis_utils import analyze_results

AZURE_OPENAI_MODEL_MAPPING = {
    'gpt-4o': os.environ["AZURE_OPENAI_GPT_4o"],
    'gpt-4o-mini': os.environ["AZURE_OPENAI_GPT_4o_mini"],
    'gpt-o3-mini': os.environ["AZURE_OPENAI_GPT_o3_mini"]
}

AZURE_AI_MODEL_MAPPING = {
    'Meta-Llama-3.1-405B-Instruct': os.environ["AZURE_AI_LLAMA_405B"]
}
    
def run_pipeline(
        benchmark,
        llm_handler,
        model,
        llm_eval_handler,
        llm_eval_model,
        bq_handler,
        schema,
        experiment,
        num_examples,
        thresholds,
        prompts_dir='prompts',
        out_dir='results',
        plots_dir='results/plots',
        rerun=True
):  
    if isinstance(llm_handler, AzureOpenAILLM):
        model_name = AZURE_OPENAI_MODEL_MAPPING.get(model, 'gpt-4o')
    elif isinstance(llm_handler, AzureAILLM):
        model_name = AZURE_AI_MODEL_MAPPING.get(model, 'Meta-Llama-3.1-405B-Instruct')
    else:
        model_name = model

    if isinstance(llm_eval_handler, AzureOpenAILLM):
        llm_eval_model_name = AZURE_OPENAI_MODEL_MAPPING.get(llm_eval_model, 'gpt-4o')
    else:
        llm_eval_model_name = llm_eval_model
    
    sql_gen_prompt, example_query_prompt, nl_answer_prompt, bioscore_prompt = read_prompts(prompt_dir=prompts_dir)

    example_queries = get_examples(example_query_prompt, num_examples)

    threshold_rules = get_thresholds(thresholds)

    columns = [
        'uuid','sql_input_tokens','sql_gen_response','sql_gen_time','parsed_sql_query','sql_returned','exec_results',
        'sql_ran','sql_exec_time','answer','answer_input_tokens','answer_gen_response','answer_gen_time','bioscore',
        'bioscore_input_tokens','bioscore_time','total_time','input_tokens'
    ]

    results_list = []

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/experiment_results', exist_ok=True)
    os.makedirs(f'{plots_dir}/experiment_plots', exist_ok=True)

    if rerun or not os.path.isfile(f'{out_dir}/experiment_results/{model}-{experiment}-results.csv'):
        for ind, row in benchmark.iterrows():
            row_result = {col: None for col in columns}
            row_result['uuid'] = row['uuid']

            question_sql_gen_prompt = sql_gen_prompt.format(
                question=row['question'],
                db_schema=schema,
                threshold_rules=threshold_rules,
                example_queries=example_queries
            )

            total_time_start = time.perf_counter()
            sql_gen_start = time.perf_counter()
            sql_query, sql_input_tokens, sql_gen_response = generate_sql(
                llm_handler=llm_handler,
                model_name=model_name,
                prompt=question_sql_gen_prompt
            )
            sql_gen_end = time.perf_counter()

            with open('test.txt', 'a+') as f:
                f.write(f"{row['question']}\n")
                f.write(f'{sql_query}\n')

            row_result['sql_input_tokens'] = sql_input_tokens
            row_result['sql_gen_response'] = sql_gen_response
            row_result['sql_gen_time'] = sql_gen_end - sql_gen_start

            if sql_gen_response == 1:
                parsed_sql_query, sql_returned = parse_sql_query(query=sql_query)

                with open(f'test.txt', 'a+') as f:
                    f.write(f"{parsed_sql_query}\n\n")

                row_result['parsed_sql_query'] = parsed_sql_query
                row_result['sql_returned'] = sql_returned

                print(f"Question: {row['question']}")
                print(f"Number of input tokens for SQL generation: {sql_input_tokens}")
                print(f"Generated SQL: {parsed_sql_query}")

                if sql_returned == 1:
                    sql_exec_start = time.perf_counter()
                    exec_results, sql_ran = run_sql(bq_handler=bq_handler, query=parsed_sql_query)
                    sql_exec_end = time.perf_counter()

                    row_result['exec_results'] = exec_results
                    row_result['sql_ran'] = sql_ran
                    row_result['sql_exec_time'] = sql_exec_end - sql_exec_start

                    print(f"Number of rows returned: {len(exec_results)}")

                    question_nl_answer_prompt = nl_answer_prompt.format(
                        question=row['question'],
                        sql_query=parsed_sql_query,
                        execution_results=exec_results
                    )
                    answer_gen_start = time.perf_counter()
                    answer, answer_input_tokens, answer_gen_response = generate_answer(
                        llm_handler=llm_handler,
                        model_name=model_name,
                        prompt=question_nl_answer_prompt
                    )
                    answer_gen_end = time.perf_counter()
                    total_time_end = time.perf_counter()

                    row_result['answer'] = answer
                    row_result['answer_input_tokens'] = answer_input_tokens
                    row_result['answer_gen_response'] = answer_gen_response
                    row_result['answer_gen_time'] = answer_gen_end - answer_gen_start
                    row_result['total_time'] = total_time_end - total_time_start
                    row_result['input_tokens'] = sql_input_tokens + answer_input_tokens

                    print(f"Number of input tokens for answer generation: {answer_input_tokens}")
                    print(f"Answer: {answer}")
                    
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

                    print(f'Bioscore: {float(bioscore)}')

            results_list.append(row_result)

        results_df = pd.DataFrame(results_list)
    else:
        results_df = pd.read_csv(f'{out_dir}/{model}-{experiment}-results.csv')

    metrics_df, results = analyze_results(
        results=results_df,
        benchmark=benchmark,
        model=model,
        experiment=experiment,
        plots_dir=f'{plots_dir}/experiment_plots'
    )

    results.to_csv(f'{out_dir}/experiment_results/{model}-{experiment}-results.csv', index=False)
    metrics_df.to_csv(f'{out_dir}/experiment_results/{model}-{experiment}-metrics.csv', index=False)

def main():
    config_path = 'config/llm_config.yaml'

    bq_handler = BigQuery(BQ_CLIENT)

    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        paths = config.get("paths", {})
        prompts_dir = paths.get("prompts_dir", "prompts")
        results_dir = paths.get("results_dir", "results")
        benchmark_path = paths.get("benchmark_path", "data/benchmark_data/CARDBiomedBenchSQL_sample500.csv")

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
            
        benchmark = benchmark.head(2)

        if eval_model_provider == 'azure_openai':
            llm_eval_handler = AzureOpenAILLM(AZURE_CLIENT)
        elif eval_model_provider == 'gemini':
            llm_eval_handler = GeminiLLM(GEMINI_CLIENT)
        else:
            raise ValueError(f'Invalid LLM Provider Passed: {model_provider}')

        if len(experiments) > 0 and len(experiment_models) > 0:
            for model in experiment_models:
                model_provider = model.get("provider")
                model_name = model.get("model")

                if model_provider == 'azure_openai':
                    llm_handler = AzureOpenAILLM(AZURE_CLIENT)
                elif model_provider == 'azure_ai':
                    llm_handler = AzureAILLM(AZURE_AI_CLIENT)
                elif model_provider == 'gemini':
                    llm_handler = GeminiLLM(GEMINI_CLIENT)
                elif model_provider == 'anthropic':
                    llm_handler = AnthropicLLM(ANTHROPIC_CLIENT)
                elif model_provider == 'huggingface':
                    model, tokenizer, device = HuggingFaceLLM.initialize_llm_client(
                        model_name=model_name,
                        auth_token=os.environ['HF_TOKEN'],
                        torch_dtype=torch.bfloat16,
                        device=None
                    )
                    llm_handler = HuggingFaceLLM(
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                    model_name = model_name.split('/')[1]
                    print(model_name)
                else:
                    raise ValueError(f'Invalid LLM Provider Passed: {model_provider}.')


                for experiment in experiments:
                    experiment_name = experiment.get("name")
                    num_rows = experiment.get("num_rows", 0)
                    num_examples = experiment.get("num_examples", 0)
                    thresholds = experiment.get("thresholds", False)

                    if experiment_name == 'baseline':
                        run_experiment = True
                    elif model_name == 'gpt-o3-mini':
                        run_experiment = True
                    else:
                        run_experiment = False

                    print(model_name, experiment_name)

                    if run_experiment:

                        schema = create_table_info(
                            bq_handler=bq_handler,
                            dataset_id='bio_sql_benchmark',
                            num_rows=num_rows
                        )

                        run_pipeline(
                            benchmark=benchmark,
                            llm_handler=llm_handler,
                            model=model_name,
                            llm_eval_handler=llm_eval_handler,
                            llm_eval_model=eval_model_name,
                            bq_handler=bq_handler,
                            schema=schema,
                            experiment=experiment_name,
                            num_examples=num_examples,
                            thresholds=thresholds,
                            prompts_dir=prompts_dir,
                            out_dir=results_dir,
                            plots_dir=f'{results_dir}/plots',
                            rerun=True
                        )

                        if model_provider == 'huggingface':
                            try:
                                del model
                            except Exception as e:
                                print('Could not delete model')
                            gc.collect()
                            torch.cuda.empty_cache()
        else:
            raise ValueError(f'No Experiment Models or Experiments Passed via config/config.yaml.')
    else:
        raise FileNotFoundError(f'Cannot find file: {config_path}.')
        
if __name__ == '__main__':
    main()