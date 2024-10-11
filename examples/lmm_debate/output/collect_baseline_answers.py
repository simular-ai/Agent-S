import os 
import json 
import pandas as pd 
from collections import Counter 

random_sels = ['pi',
 'DC_metro',
 'phone_sales',
 'population growth',
 'simpson',
 'math_prob',
 'population',
 'china_export_us',
 'teen_population',
 'para_angle',
 'teen_population',
 'math_prob',
 'world_war2',
 'sqrt2',
 'duck',
 'teen_population',
 'line',
 'illusion',
 'Red Velvet',
 'math_prob',
 'usmap',
 'central_bank',
 'flow',
 'line',
 'circle',
 'square',
 'Kennedy',
 'Berlin',
 'NBA',
 'parking']

def open_log_file(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def collect_date_split_answers():    
    debate_baseline_31st_file = '/home/saaket/lmm-agents/logs/2024_31_5/baseline_answers_20240530_200504.json'
    debate_31st_file = '/home/saaket/lmm-agents/logs/2024_31_5/answers_debate_20240530_200504.json'
    
    actor_critic_30th_file = '/home/saaket/lmm-agents/logs/2024_30_5/answers_debate_20240530_200754.json'

    actor_critic = open_log_file(actor_critic_30th_file)

    debate_baseline_31st = open_log_file(debate_baseline_31st_file)
    debate_31st = open_log_file(debate_31st_file)
    debate = debate_31st
    debate_baseline = debate_baseline_31st
    assert len(debate) == len(debate_baseline) == len(actor_critic)
    return debate, debate_baseline, actor_critic




def full_debate_compile(answers):
    
    # df = pd.read_json('/home/saaket/lmm-agents/lmm_debate/got_these_wrong_before_vd.json')
    # df = df.reset_index() 

    # df = df.iloc[:240]
    hallusion_path = '/data4/saaket/hallusion_bench/'
    data_file = os.path.join(hallusion_path, 'HallusionBench.json')
    with open(data_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    df = df[df['sample_note'].isin(random_sels)]

    baseline_answers = open_log_file(answers['baseline'])
    debate_answers = open_log_file(answers['debate'])

    baseline_answers_df = df.copy()
    debate_answers_df = df.copy()

    baseline_answers_df['model_prediction'] = baseline_answers
    debate_answers_df['model_prediction'] = debate_answers

    # make dirs
    os.makedirs('output/full_debate_corrected_mod_prompt/debate_baseline', exist_ok=True)
    os.makedirs('output/full_debate_corrected_mod_prompt/debate', exist_ok=True)

    debate_answers_df.to_json('output/full_debate_corrected_mod_prompt/debate/HallusionBench_result.json', orient='records')
    baseline_answers_df.to_json('output/full_debate_corrected_mod_prompt/debate_baseline/HallusionBench_result.json', orient='records')


def compile_in_hallusion_format_old():
    debate, debate_baseline, actor_critic = collect_date_split_answers()
    
    # Load the hallusionbench data
    hallusion_path = '/data4/saaket/hallusion_bench/'
    data_file = os.path.join(hallusion_path, 'HallusionBench.json')
    with open(data_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(df)
    #### DATA FILTERING #####

    # What is the size of the df and the distribution of categories and subcategories
    print(len(df))
    print(Counter(df['category']))
    print(Counter(df['subcategory']))


    # Let's only keep the visual dependent examples (VD)
    df_vd = df.loc[df['category'] == 'VD'].reset_index(drop=True)
    print(df_vd)
    # Create three new dataframed debate_answers, debate_baseline_answers, and actor_critic_answers
    # Each datafram is df_vd with a new column with the key model_prediction which will correspond to the corresponding answers
    # from the debate, debate_baseline, and actor_critic lists
    debate_answers = df_vd.copy()
    debate_baseline_answers = df_vd.copy()
    actor_critic_answers = df_vd.copy()
    
    debate_answers['model_prediction'] = debate
    debate_baseline_answers['model_prediction'] = debate_baseline
    actor_critic_answers['model_prediction'] = actor_critic

    # Save the three dataframes as json files named HallusionBench_result.json in three different folders
    # Name the folders debate, debate_baseline, and actor_critic in the current directory, make folders if they dont exist
    os.makedirs('debate', exist_ok=True)
    os.makedirs('debate_baseline', exist_ok=True)
    os.makedirs('actor_critic', exist_ok=True)
    
    debate_answers.to_json('debate/HallusionBench_result.json', orient='records')
    debate_baseline_answers.to_json('debate_baseline/HallusionBench_result.json', orient='records')
    actor_critic_answers.to_json('actor_critic/HallusionBench_result.json', orient='records')

    print("Saved the three dataframes as json files named HallusionBench_result.json in three different folders")


def collect_baseline_answers():
    logs = os.listdir('/home/saaket/lmm-agents/logs/2024_30_5')

    for log in logs:
        if log.startswith("answers"):
            logs.remove(log)
        elif log.startswith("baseline"):
            logs.remove(log)


    baseline_answers = []
    for i, log in enumerate(logs):
        with open(f'/home/saaket/lmm-agents/logs/2024_30_5/{log}', 'r') as f:
            data = json.load(f)
        print(data)
        baseline_answers.append(data[1].replace("My answer: ", ''))

    with open(f'/home/saaket/lmm-agents/logs/2024_30_5/answers_baseline.json', 'w') as f:
        json.dump(baseline_answers, f)

        
if __name__ == '__main__':
    # collect_baseline_answers()
    # print(collect_date_split_answers()[0][:5])
    # print("Collected baseline answers.
    # compile_in_hallusion_format()
    full_debate_compile(
        {'baseline': '/home/saaket/lmm-agents/logs/2024_10_6/baseline_answers_20240610_135648.json',
         'debate': '/home/saaket/lmm-agents/logs/2024_10_6/answers_debate_20240610_135648.json'}
    )