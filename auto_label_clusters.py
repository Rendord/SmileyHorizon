from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import rcParams
#from sklearn.manifold import TSNE
import umap
import hdbscan
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
import json

# --- Config
NUM_EXAMPLES = 2
NUM_CLUSTERS = 40  
MODEL = "mistral"  
URL = "http://localhost:11434/api/generate"

# Font Family for Plot:
rcParams['font.family'] = 'Apple Color Emoji'

emoji_names = []

# --- Step 1: Load your emoji names
with open("input/emoji_names.txt", "r") as f:
    emoji_names = [line.strip().replace('-', ' ').lower() for line in f if line.strip()]

# --- Step 2: Embed the names
model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(emoji_names)

# # --- Step 3: Cluster the embeddings
# kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
# labels = kmeans.fit_predict(embeddings)

# --- Step 3: Cluster the embeddings dynamically
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0, metric='euclidean')
labels = clusterer.fit_predict(embeddings)

# --- Step 4: Organize by cluster
clusters = defaultdict(list)
for name, label in zip(emoji_names, labels):
    if label != -1:  # ignore noise points
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

# # --- Step 6: Print result
# for cid, names in sampled_clusters.items():
#     print(f"\n### Cluster {cid} ({cluster_labels.get(cid, 'Unknown')}):")
#     print(", ".join(names))  # top 10 per cluster
#     top_picks.extend(names)

# --- Step 6: Print result with ALL entries in each cluster
with open('output/top_picks_40_clusters.txt', 'w') as f:
    for cid, names in clusters.items():
        print(f"\n### Cluster {cid} ({cluster_labels.get(cid, 'Unknown')}):")
        print(", ".join(names))  # print all names in this cluster

        # also write them to file
        for name in names:
            f.write(name + "\n")

with open('output/top_picks_40_clusters.txt', 'w') as f:
        f.writelines(name + "\n" for name in top_picks)


# Load the Dutch-to-emoji mapping
with open('output/dutch_to_emoji.json', 'r', encoding='utf-8') as f:
    dutch_to_emoji = json.load(f)

# Build a list of emoji characters matching your original emoji_names
emoji_chars = [dutch_to_emoji.get(name, "❓") for name in emoji_names]


# If you have HDBSCAN labels, some might be -1 (noise); filter them out
valid_indices = [i for i, lbl in enumerate(labels) if lbl != -1]
filtered_embeddings = np.array([embeddings[i] for i in valid_indices])
filtered_labels = [labels[i] for i in valid_indices]
filtered_emojis = [emoji_chars[i] for i in valid_indices]  # <-- actual emoji characters

#reduce embeddings to 2D
reducer = umap.UMAP(random_state=42)
embeddings_2d = reducer.fit_transform(filtered_embeddings)

x = embeddings_2d[:, 0]
y = embeddings_2d[:, 1]




# #Create plot
# plt.figure(figsize=(16, 12))
# for xi, yi, emoji_char, lbl in zip(x, y, filtered_emojis, filtered_labels):
#     plt.text(xi, yi, emoji_char, fontsize=14)  # directly plot emoji

# plt.xticks([])
# plt.yticks([])
# plt.title("Emoji Clusters in 2D (umap projection)", fontsize=18)
# plt.tight_layout()
# plt.show()

# Assume you already have:
# x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]
# filtered_emojis = [emoji_chars[i] for i in valid_indices]

# Build DataFrame
df = pd.DataFrame({
    "x": embeddings_2d[:, 0],
    "y": embeddings_2d[:, 1],
    "emoji": filtered_emojis
})

#Debug peek
print(df.head())

fig = go.Figure()

# Scatter with emoji text
fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='text',
    text=df['emoji'],
    textfont=dict(size=30),
    name='emoji'
))

# # Scatter with normal labels (index as fallback)
# fig.add_trace(go.Scatter(
#     x=df['x'],
#     y=df['y'],
#     mode='text',
#     text=[str(i) for i in range(len(df))],
#     textfont=dict(size=20, color='red'),
#     name='index labels'
# ))
# Save and open in browser
fig.write_html("output/emoji_clusters.html", auto_open=True)



