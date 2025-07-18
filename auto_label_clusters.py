from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
import requests
import pandas as pd
import random

# --- Config
NUM_EXAMPLES = 2
NUM_CLUSTERS = 40  
MODEL = "mistral"  
URL = "http://localhost:11434/api/generate"

emoji_names = []

# --- Step 1: Load your emoji names
with open("input/auto_label_clusters/emoji_names.txt", "r") as f:
    emoji_names = [line.strip() for line in f if line.strip()]

# --- Step 2: Embed the names
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(emoji_names)

# --- Step 3: Cluster the embeddings
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(embeddings)

# --- Step 4: Organize by cluster
clusters = defaultdict(list)
for name, label in zip(emoji_names, labels):
    clusters[label].append(name)

def process_output(text):
    for ch in [",", ".", "!", "?", ":", ";", "&", "/", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, " ")

    tokens = text.strip().split()
    
    return tokens[0].lower() if tokens else ''

def get_random(list, size):
    return random.sample(list, min(len(list), size))


# --- Optional Step 5: Auto-label clusters using GPT
def label_cluster(names):   

    prompt = (
                "You are an expert summarizer.\n "
                "Name this group of emoji names in one word.\n" 
                "Constraints:\n"
                "NO list or explanation.\n"
                "NO individual emoji names.\n"
                "RESPOND ONLY with a concise label (like 'Food' or 'Animals' or 'Faces') even if not accurate.\n"
                "If categories don't align the label becomes the one with the highest frequency.\n\n"
                f"Emoji names: {', '.join(names)}\n\n"
                "Your answer:"
    )
    
    json = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 2,
        "temperature": 0.2,
        "stream": False
    }

    response = requests.post(URL, json=json)


    return process_output(response.json()['response'])


sampled_clusters = {
    cid: get_random(names, NUM_EXAMPLES)
    for cid, names in clusters.items()
}


cluster_labels = {
    cid: label_cluster(names) for cid, names in sampled_clusters.items()
}


top_picks = []

# --- Step 6: Print result
for cid, names in sampled_clusters.items():
    print(f"\n### Cluster {cid} ({cluster_labels.get(cid, 'Unknown')}):")
    print(", ".join(names))  # top 10 per cluster
    top_picks.extend(names)

with open('output/auto_label_clusters/top_picks_40_clusters.txt', 'w') as f:
        f.writelines(name + "\n" for name in top_picks)