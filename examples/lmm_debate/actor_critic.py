# Author: Saaket Agashe
# Date: 2021-09-15
# License: MIT

from lmm_agents.MultimodalAgent import LMMAgent
import json 
import pandas as pd
from collections import Counter
from tqdm import tqdm 
from paths import LOG_PATH, DATA_PATH
from datetime import datetime 
import os 
# Debate about a question about an image in the hallusion bench dataset 
def actor_critic(identifier, n_rounds, actor, critic, prompt, image_path):
    
    # Check all the required agents are provided
    if not actor or not critic:
        raise ValueError("actor-critic requires both actor and critic agents to be provided")

    conversation_history = []
    initial_aff_agent_response = None 
    for round in range(n_rounds):
        # Start the debate 
        # Only need to pass the image and set the tone of the conversation (disagreement) in the first turn 
        if round == 0:
            # The image and question is passed as input to the affirmative agent
            actor.add_message(prompt + " Think step by step.", image_content=image_path)
            conversation_history.append(f"Question: {prompt}")
            response = actor.get_response()
            conversation_history.append(f"My answer: {response}")
            actor.add_message(response)


            critic_prompt = f"{prompt}. My answer: {response}. Check if my reasoning is correct. think step by step."
            critic.add_message(critic_prompt, image_content=image_path)
            critic_response = critic.get_response()
            conversation_history.append(f"Critic: {critic_response}")
            critic.add_message(critic_response)

            if 'Verification: Success'.lower() in critic_response.lower() or 'sucess' in critic_response.lower():
                break
        else:
            # The negative agent's response is passed to the affirmative agent
            actor.add_message(critic_response)
            response = actor.get_response()
            conversation_history.append(f"My answer: {response}")
            actor.add_message(response)


            critic_prompt = f"{prompt}. My updated answer: {response}. Check if my reasoning is correct. Think step by step."
            critic.add_message(critic_prompt, image_content=image_path)
            critic_response = critic.get_response()
            conversation_history.append(f"Critic: {critic_response}")
            critic.add_message(critic_response)

            if 'Verification: Success'.lower() in critic_response.lower() or 'sucess' in critic_response.lower():
                break
            
            
    # extract everything after Final Answer from moderator_response 
    final_answer = response 

    # Save conversation history in a json log file 
    today = datetime.today()
    year, day, month = today.year, today.day, today.month

    # makedirs with name blah if they don't exist 
    os.makedirs(f"{LOG_PATH}/{year}_{day}_{month}", exist_ok=True)

    with open(f"{LOG_PATH}/{year}_{day}_{month}/actor_critic_history_{identifier}.json", "w") as f:
        json.dump(conversation_history, f)

    return final_answer
        
    
