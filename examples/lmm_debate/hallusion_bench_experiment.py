# Author: Saaket Agashe
# Date: 2021-09-15
# License: MIT

# Standard Imports 
from collections import Counter 
import json 
from tqdm import tqdm
import os 
import pandas as pd 
from datetime import datetime
from PIL import Image, ImageDraw

# Custom Imports 
from debate import debate, debate2
from actor_critic import actor_critic
from lmm_agents.MultimodalAgent import LMMAgent
from prompts import DEBATER_SYSTEM_PROMPT, MODERATOR_SYSTEM_PROMPT, ACTOR_PROMPT, CRITIC_PROMPT, MODERATOR_FINAL_PROMPT
from paths import LOG_PATH, DATA_PATH
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
)
# Set to true for debate before answering 


experiment_config = {
    'strategy': 'debate', # debate, actor_critic, baseline
    'n_rounds': 2,
    'model_type': 'azure'
}

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



def visualize():
        # Load the hallusionbench data
    hallusion_path = '/data4/saaket/hallusion_bench/'
    data_file = os.path.join(hallusion_path, 'HallusionBench.json')
    with open(data_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    #### DATA FILTERING #####

    # What is the size of the df and the distribution of categories and subcategories
    print(len(df))
    print(Counter(df['category']))
    print(Counter(df['subcategory']))


    # Let's only keep the visual dependent examples (VD)
    df_vd = df.loc[df['category'] == 'VD']

    # This code will sample n_samples examples from each subcategory
    n_samples = 5
    random_seed = 42  # Set the seed for reproducibility

    df_vd_illusion_only = df_vd[df_vd['subcategory'] == 'illusion']
    df_vd_illusion_selected_sample_keys = df_vd_illusion_only[df_vd_illusion_only['sample_note'].isin(['circle', 'box', 'line', 'rail', 'grey_dot'])]
    sample_df = df_vd_illusion_selected_sample_keys.groupby('sample_note', group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples), random_state=random_seed))


    #### VISUALIZATION #####
    answers = []
    sample_df = sample_df.reset_index()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for index, row in tqdm(sample_df.iterrows()):
        # print(row['question'])
        # if index == 20:
        #     break 
        question = row['question']
        filename = row['filename']
        image_path = hallusion_path + filename
        
        # Open image, write the question on the bottom of the image and save
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), question, (255, 255, 255))
        img.save(f"{LOG_PATH}/{question}_{index}.png")

def main():

    # Load the hallusionbench data
    hallusion_path = '/data4/saaket/hallusion_bench/'
    data_file = os.path.join(hallusion_path, 'HallusionBench.json')
    # data_file = 'got_these_wrong_before_vd.json'
    with open(data_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    #### DATA FILTERING #####

    # What is the size of the df and the distribution of categories and subcategories
    print(len(df))
    print(Counter(df['category']))
    print(Counter(df['subcategory']))


    # Let's only keep the visual dependent examples (VD)
    # df_vd = df.loc[df['category'] == 'VD']
    # df_vs = df.loc[df['category'] == 'VS']
    # # This code will sample n_samples examples from each subcategory
    # n_samples = 5
    # random_seed = 42  # Set the seed for reproducibility

    # df_vd_illusion_only = df_vd[df_vd['subcategory'] == 'illusion']
    # print(Counter(df_vd_illusion_only['sample_note']))
    # df_vd_illusion_selected_sample_keys = df_vd_illusion_only[df_vd_illusion_only['sample_note'].isin(['circle', 'box', 'line', 'rail', 'grey_dot'])]
    # df_vd_illusion_selected_sample_keys = df_vd_illusion_only[~df_vd_illusion_only['sample_note'].isin(['circle', 'line'])]
    # # sample_df = df_vd_illusion_selected_sample_keys.groupby('sample_note', group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples), random_state=random_seed))
    # print(Counter(df_vd_illusion_selected_sample_keys['sample_note']))
    # sample_df = df_vd_illusion_selected_sample_keys
    sample_df = df[df['sample_note'].isin(random_sels)]
    print(sample_df.head())
    print(len(sample_df))
    print(Counter(sample_df['category']))
    #### AGENT INITIALIZATION #####
    if experiment_config['model_type'] == 'llava':
        engine_params = {
            "engine_type": 'llava',
            'model_path': 'liuhaotian/llava-v1.5-7b',
        }
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
                engine_params['model_path'], None, get_model_name_from_path(engine_params['model_path']))
        engine_params['tokenizer'] = tokenizer 
        engine_params['model'] = model 
        engine_params['image_processor'] = image_processor
        engine_params['context_len'] = context_len
   
    elif experiment_config['model_type'] == 'cogvlm':
        engine_params = {
            'engine_type': 'cogvlm',
            'model_path': "THUDM/cogvlm2-llama3-chat-19B"
        }
        
    else:
        engine_params = {
            "engine_type": "azure",
            "model": "guiagents",
            "api_version": "2023-12-01-preview"
        }

    #### AGENT WORKFLOW #####
    answers = []
    baseline_answers = []
    sample_df = sample_df.reset_index()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    
    for index, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0]):
        # print(row['question'])
        # if index == 20:
        #     break 
        # if index < 1066:
        #     continue
        question = row['question']
        filename = row['filename']
        if filename:
            image_path = hallusion_path + filename
        else:
            image_path = None 
        # Log the values to visualize what is happening
        print(f"Generating answer for question: {question}")
        print(f"Using image path: {image_path}")
        
        # Generate the answer using the engine
        
        if experiment_config['strategy'] == 'debate':
            identifier = f"{index}_{question[:10]}"
            aff_agent = LMMAgent(engine_params=engine_params, system_prompt=DEBATER_SYSTEM_PROMPT)
            neg_agent = LMMAgent(engine_params=engine_params, system_prompt=DEBATER_SYSTEM_PROMPT)
            moderator = LMMAgent(engine_params=engine_params, system_prompt=MODERATOR_FINAL_PROMPT)
            answer, baseline_answer = debate2(identifier=identifier,
                         n_rounds=experiment_config['n_rounds'],
                         aff_agent=aff_agent, 
                         neg_agent=neg_agent,
                         moderator=moderator,
                         prompt=question,
                         image_path=image_path)
        elif experiment_config['strategy'] == 'actor_critic':
            identifier = f"{index}_{question[:10]}"
            actor = LMMAgent(engine_params=engine_params, system_prompt=ACTOR_PROMPT)
            critic = LMMAgent(engine_params=engine_params, system_prompt=CRITIC_PROMPT)
            answer = actor_critic(identifier=identifier, 
                                  n_rounds=2,
                                  actor=actor, 
                                  critic=critic, 
                                  prompt=question, 
                                  image_path=image_path)
        else:
            agent = LMMAgent(engine_params=engine_params, system_prompt=DEBATER_SYSTEM_PROMPT)
            agent.add_message(question, image_content=image_path)
            answer = agent.get_response()
        
        
        # Log the generated answer
        print(f"Generated answer: {answer}")

        # Collect the generated answer 
        answers.append(answer)
        
        if experiment_config['strategy'] == 'debate':
            baseline_answers.append(baseline_answer)

        # Save the answers to a unique file

        # Save conversation history in a json log file 
        today = datetime.today()
        year, day, month = today.year, today.day, today.month

        # makedirs with name blah if they don't exist 
        os.makedirs(f"{LOG_PATH}/{year}_{day}_{month}", exist_ok=True)
        
        with open(f'{LOG_PATH}/{year}_{day}_{month}/answers_debate_{timestamp}.json', 'w') as f:
            json.dump(answers, f)

        with open(f'{LOG_PATH}/{year}_{day}_{month}/baseline_answers_{timestamp}.json', 'w') as f:
            json.dump(baseline_answers, f)

if __name__ == '__main__':
    main()
    # visualize()
