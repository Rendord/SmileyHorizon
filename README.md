# Emoji Name Clustering for NLU Template Optimization

## Embedding and grouping Similar Names with Clustering

**To run:**
- install the requirements in a conda environment specifically*
- run ollama on port 11434
- activate the environment and run auto_label_clusters.py

*you'll most likely need conda for it's precompiled binaries YMMV

### Goal
Group the embedding vectors so that emoji names with similar meanings are clustered together. This allows us to identify semantic categories and reduce redundancy in our dataset.

### Clustering Approach
K-Means and DBSCAN clustering to group the emoji name embeddings. Each cluster represents a semantic category where similar emoji names are grouped together.

**What the clusters represent:**
- One cluster might contain various names for facial expressions (happy, sad, smiling)
- Another cluster could group animals (cat, dog, elephant)
- A third might contain hand gestures (thumbs up, peace sign, waving)
- Additional clusters for objects, food items, activities, etc.

### Clustering Results
The algorithm successfully identified distinct semantic groups, creating meaningful categories that align with human intuition about emoji relationships.

## Understanding the Clusters with an LLM

### Goal
Generate human-readable summaries of what each cluster represents to help validate our clustering approach and assist in selecting representative emoji names.

### LLM Integration with Ollama
**Mistral** running locally on **Ollama** to analyze and summarize our clusters.

### Prompting Strategy
**The Challenge:** Mistral proved to be "a bit headstrong," requiring repetitive prompting to produce concise, consistent outputs.

**Approach:**
1. Feed 5-10 random emojis from each cluster to the LLM
2. Ask it to summarize the category with prompts like:
```
"Here are some emoji names: [happy_face, smiling, grinning, cheerful]. 
What is the main theme or category of these emojis? 
Provide a brief, one-word category label."
```


### Output Processing
Cleaned and standardized the LLM outputs to produce consistent labels for our clusters, ensuring uniformity across all categories.

### Value Added
The LLM helped:
- Quickly understand the semantic space covered by emoji names
- Validate that our clustering approach was working correctly
- Generate meaningful category labels for downstream processing

## Step 4: Selecting the Final Payload

### Selection Strategy
Approach for choosing representative names from each cluster:

- **Representative Sampling:** Pick 1-2 representative names from each cluster
- **Cluster Size Priority:** Give preference to names from larger, more significant clusters
- **Manual Review:** Manually review cluster contents to ensure diverse examples
- **Semantic Coverage:** Ensure the final selection covers the full range of emoji categories

### Final Payload
The process resulted in a significantly reduced, yet semantically comprehensive set of emoji names that maintained coverage across all major categories.

## The Outcome: Bug Squashed (and a Cool Tool Built)

### Looking forward

1. **Better Representation Selection**
- Instead of random sampling, select examples based on their diversity within clusters
- Use centroid-based selection for more representative examples

2. **Dynamic Cluster Discovery**
- Find optimal number of clusters dynamically instead of predetermined amounts
- Use techniques like elbow method or silhouette analysis

3. **Cost vs. Quality Trade-offs**
- Balance investment in optimization against eventual quality improvements
- Consider diminishing returns of additional refinement

### Future Applications
This approach could be extended to:
- **Cleaning other messy lists** in NLU datasets
- **Generating smaller, representative test sets** for model evaluation
- **Understanding semantic coverage** of existing templates
- **Optimizing training data** for various NLU tasks

### Challenges Faced
- **LLM Consistency:** Getting Mistral to produce uniform, concise outputs required significant prompt iteration
- **Cluster Validation:** Ensuring clusters made semantic sense required human review
- **Balance:** Finding the right trade-off between dataset size and semantic coverage


This project demonstrates how modern AI tools can be combined creatively to solve practical engineering problems. The intersection of embeddings, clustering, and language models opens up numerous possibilities for improving NLU systems and data processing workflows.

---

*Thank you for reading! This project showcases how sometimes the most satisfying solutions come from applying well-known techniques in novel ways to solve specific, real-world problems.*
