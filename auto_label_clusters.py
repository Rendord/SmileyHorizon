from __future__ import annotations
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
from pathlib import Path
import re
import shutil
from dataclasses import dataclass

def normalize_filename(name: str) -> str:
    name = re.sub(r'[\\/:"*?<>|]+', "", name)
    name = name.strip(" .")
    return name if name else "untitled"

# --- Config
NUM_EXAMPLES = 10
NUM_CLUSTERS = 40  
MODEL = "mistral"  
URL = "http://localhost:11434/api/generate"

# --- Cluster Distance Scaling
BASE_DISTANCE = 1.0

# Font Family for Plot:
rcParams['font.family'] = 'Apple Color Emoji'

emoji_names = []


@dataclass
class DataPoint:
    emoji_name: str
    vector: np.ndarray
    cluster_id: int

class Cluster:
    datapoints: list
    centroid: list
    cluster_id: int

    def __init__(self, datapoints: list[np.ndarray]):
        self.datapoints = datapoints
        self.centroid = calculate_centroid(datapoints)
    
    def update_centroid(self):
        self.centroid = np.mean(self.datapoints, axis=0)

    def __iadd__(self, force: ClusterForce):
        self.centroid += force.force
        self.datapoints = [dp + force.force for dp in self.datapoints]

        return self

@dataclass
class ClusterForce:
    cluster: Cluster
    force: np.ndarray


# --- Step 1: Load your emoji names
with open("input/emoji_names.txt", "r") as f:
    emoji_names = [line.strip().replace('-', ' ').lower() for line in f if line.strip()]

#TODO!!!!! Finetune Transformers on Emoji dataset
# --- Step 2: Embed the names
model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(emoji_names)


# # --- Step 3: Cluster the embeddings -- KMeans Example (Deprecated)
# kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
# labels = kmeans.fit_predict(embeddings)

def calculate_scalar(centroid: np.ndarray, clusters: list[Cluster]) -> float:
    distances = []
    for c in clusters:
        diff_vector = centroid - c.centroid
        distance = np.linalg.norm(diff_vector)
        distances.append(distance)
    avg_distance = sum(distances) / len(distances)
    scalar = BASE_DISTANCE / avg_distance
    return scalar

def calculate_cluster_separation_force(centroid: np.ndarray, clusters: list[Cluster]) -> np.ndarray:
    scalar = calculate_scalar(clusters)
    total_force = np.array()
    for c in clusters:
        if c.centroid == centroid:
            pass
        else: 
            c_diff = centroid - c.centroid
            normalized_diff = c_diff / np.linalg.norm(c_diff)
            force = centroid + (normalized_diff * scalar)
            total_force += force
    return total_force

def seperate_clusters(clusters: list[Cluster]) -> None:
    cluster_forces = [ClusterForce(cluster, calculate_cluster_separation_force(cluster. centroid, clusters)) for cluster in clusters]
    for cf in cluster_forces:
        cf.cluster += cf.force

def get_cluster_vectors(cluster_id):
    return list

def calculate_closest_centroid(centroids: list, outlier):
    if not centroids:
        return None
    closest =  0

    return closest


def calculate_centroid(vectors: list):
    if not vectors:
        return None
    sum_of_vectors = sum(v for v in vectors)
    num_vectors = len(vectors)
    centroid_vector = (1 / num_vectors) * sum_of_vectors
    return centroid_vector

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

#TODO implement alternative service for labeling?
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

#TODO Implement more representative sampling, calculate cosine similarity between example and cluster?
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

total_names = 0
total_clusters = 0
# --- Step 6: Write clusters to files and print them to terminal
with open(f'output/top_picks_{str(NUM_CLUSTERS)}_clusters.txt', 'w') as f:
    folder = Path(f'output/clusters_{str(NUM_CLUSTERS)}')
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for cid, names in clusters.items():
        total_clusters += 1
        cluster_name = cluster_labels.get(cid, 'Unknown')
        print(f"\n### Cluster {cid} ({cluster_name}):")
        print(", ".join(names))  # print all names in this cluster

        file_path = folder / normalize_filename(f'cluster_{cluster_name}_{str(total_clusters)}.txt')
        with open(file_path, 'w') as individual_cluster:
            for name in names:
                total_names += 1
                individual_cluster.write(name + "\n")

    #write samples of clusters to file #TODO Why am I doing this?
    f.writelines(name + "\n" for name in top_picks)     

print("total_names: " + str(total_names))
print("total_clusters: " + str(total_clusters))

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



