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


def uninterrupted_debate(identifier, n_rounds, aff_agent, neg_agent, moderator, prompt, image_path):
    # Check all the required agents are provided
    if not aff_agent or not neg_agent or not moderator:
        raise ValueError("debate requires an affirmative agent, a negative agent and a moderator to be provided")

    engine_params = {
        "engine_type": "azure",
        "model": "gui-agents",
    }

    conversation_history = []
    initial_aff_agent_response = None 
    for round in range(n_rounds):
        # Start the debate 
        # Only need to pass the image and set the tone of the conversation (disagreement) in the first turn 
        if round == 0:
            # The image and question is passed as input to the affirmative agent
            aff_agent.add_message(prompt + "", image_content=image_path)
            conversation_history.append(f"Question: {prompt}")
            aff_response = aff_agent.get_response()
            conversation_history.append(f"Affirmative Agent: {aff_response}")
            aff_agent.add_message(aff_response)

            initial_aff_agent_response = aff_response

            # The affirmative agent's response is passed to the negative agent
            neg_agent_input = f"{prompt}. My response: {aff_response}. You disagree with me. State your reasons and your answer."
            neg_agent.add_message(neg_agent_input, image_content=image_path)
            neg_response = neg_agent.get_response()
            conversation_history.append(f"Negative Agent: {neg_response}")
            neg_agent.add_message(neg_response)
        else:
            # The negative agent's response is passed to the affirmative agent
            aff_agent.add_message(neg_response + "")
            aff_response = aff_agent.get_response()
            conversation_history.append(f"Affirmative Agent: {aff_response}")

            # The affirmative agent's response is passed to the negative agent
            neg_agent.add_message(f"{aff_response}. You disagree with me. State your reasons and your answer.")
            neg_response = neg_agent.get_response()
            conversation_history.append(f"Negative Agent: {neg_response}")

            if round == n_rounds-1:
                moderator_message = "\n".join(conversation_history).replace("Negative Agent", "Debater 2").replace("Affirmative Agent", "Debater 1")
                
                moderator.add_message(f"{moderator_message}. You must decide an answer this round since its the last round of debate.")
                moderator_response = moderator.get_response()
                # moderator.add_message(moderator_response)
                
                conversation_history.append(f"Moderator: {moderator_response}")
    # extract everything after Final Answer from moderator_response 
    if 'Final Answer' in moderator_response:
        final_answer = moderator_response.split("Final Answer:")[-1]
        final_answer = final_answer.strip()
    else:
        final_answer = moderator_response 

    # Save conversation history in a json log file 
    today = datetime.today()
    year, day, month = today.year, today.day, today.month

    # makedirs with name blah if they don't exist 
    os.makedirs(f"{LOG_PATH}/{year}_{day}_{month}", exist_ok=True)

    with open(f"{LOG_PATH}/{year}_{day}_{month}/conversation_history_{identifier}.json", "w") as f:
        json.dump(conversation_history, f)

    return final_answer, initial_aff_agent_response


# Debate about a question about an image in the hallusion bench dataset 
def debate(identifier, n_rounds, aff_agent, neg_agent, moderator, prompt, image_path):
    
    # Check all the required agents are provided
    if not aff_agent or not neg_agent or not moderator:
        raise ValueError("debate requires an affirmative agent, a negative agent and a moderator to be provided")

    engine_params = {
        "engine_type": "azure",
        "model": "gui-agents",
    }

    conversation_history = []
    initial_aff_agent_response = None 
    for round in range(n_rounds):
        # Start the debate 
        # Only need to pass the image and set the tone of the conversation (disagreement) in the first turn 
        if round == 0:
            # The image and question is passed as input to the affirmative agent
            aff_agent.add_message(prompt + "", image_content=image_path)
            conversation_history.append(f"Question: {prompt}")
            aff_response = aff_agent.get_response()
            conversation_history.append(f"Affirmative Agent: {aff_response}")
            aff_agent.add_message(aff_response)

            initial_aff_agent_response = aff_response

            # The affirmative agent's response is passed to the negative agent
            neg_agent_input = f"{prompt}. My response: {aff_response}. You disagree with me. State your reasons and your answer."
            neg_agent.add_message(neg_agent_input, image_content=image_path)
            neg_response = neg_agent.get_response()
            conversation_history.append(f"Negative Agent: {neg_response}")
            neg_agent.add_message(neg_response)

            moderator.add_message(f"{prompt}. Debater 1 argues: {aff_response}. Debater 2 argues: {neg_response}.", image_content=image_path)
            moderator_response = moderator.get_response()
            moderator.add_message(moderator_response)
            conversation_history.append(f"Moderator: {moderator_response}")

            if 'Final Answer' in moderator_response:
                break
        else:
            # The negative agent's response is passed to the affirmative agent
            aff_agent.add_message(neg_response + "")
            aff_response = aff_agent.get_response()
            conversation_history.append(f"Affirmative Agent: {aff_response}")

            # The affirmative agent's response is passed to the negative agent
            neg_agent.add_message(f"{aff_response}. You disagree with me. State your reasons and your answer.")
            neg_response = neg_agent.get_response()
            conversation_history.append(f"Negative Agent: {neg_response}")

            if round == n_rounds-1:
                moderator.add_message(f"{prompt}. Debater 1 argues: {aff_response}. Debater 2 argues: {neg_response}. You must decide an answer this round since its the last round of debate.")
            else:
                moderator.add_message(f"{prompt}. Debater 1 argues: {aff_response}. Debater 2 argues: {neg_response}.")

            moderator_response = moderator.get_response()
            moderator.add_message(moderator_response)
            conversation_history.append(f"Moderator: {moderator_response}")

            if 'Final Answer' in moderator_response:
                break
    # extract everything after Final Answer from moderator_response 
    final_answer = moderator_response.split("Final Answer:")[-1]
    final_answer = final_answer.strip()

    # Save conversation history in a json log file 
    today = datetime.today()
    year, day, month = today.year, today.day, today.month

    # makedirs with name blah if they don't exist 
    os.makedirs(f"{LOG_PATH}/{year}_{day}_{month}", exist_ok=True)

    with open(f"{LOG_PATH}/{year}_{day}_{month}/conversation_history_{identifier}.json", "w") as f:
        json.dump(conversation_history, f)

    return final_answer, initial_aff_agent_response
        
    
