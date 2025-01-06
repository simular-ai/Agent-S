import json
import pickle
import os
from tqdm import tqdm
import time
import openai
from openai import OpenAI


def process_embeddings_linearly(keys, model="text-embedding-3-small"):
    """
    Process embeddings one at a time using OpenAI API with basic crash recovery
    """
    client = OpenAI()  # Will use OPENAI_API_KEY from environment

    # Load any existing progress
    existing_embeddings = {}
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            existing_embeddings = pickle.load(f)
        print(f"Found {len(existing_embeddings)} existing embeddings")

    embedding_dict = existing_embeddings.copy()

    # Filter out keys that are already processed
    remaining_keys = [k for k in keys if k not in embedding_dict]
    print(
        f"Processing {len(remaining_keys)} remaining keys out of {len(keys)} total keys"
    )

    # Process one at a time with progress bar
    with tqdm(total=len(remaining_keys)) as pbar:
        for key in remaining_keys:
            # Get single embedding from OpenAI
            response = client.embeddings.create(input=key, model=model)
            embedding = response.data[0].embedding
            # time.sleep(2.0)

            # Update embedding dictionary
            embedding_dict[key] = embedding

            # Save after each embedding
            with open("embeddings.pkl", "wb") as f:
                pickle.dump(embedding_dict, f)

            pbar.update(1)

    return embedding_dict


# Load knowledge bases
with open(
    "/Users/saaketagashe/Documents/agent_s_workspace/Agent-S/agent_s/kb/osworld/lifelong_learning_knowledge_base.json"
) as f:
    ll_kb = json.load(f)

with open(
    "/Users/saaketagashe/Documents/agent_s_workspace/Agent-S/agent_s/kb/osworld/subtask_experience_knowledge_base.json"
) as f:
    subtask_kb = json.load(f)

# Combine keys
keys = list(ll_kb.keys()) + list(subtask_kb.keys())
print(f"Total number of keys to process: {len(keys)}")

# Process embeddings linearly
embedding_dict = process_embeddings_linearly(keys)

# Verify all embeddings were processed
assert len(keys) == len(
    embedding_dict
), f"Missing embeddings: {len(keys) - len(embedding_dict)} keys not processed"
print("All embeddings processed and saved successfully!")
