import os, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import sys

dataset = sys.argv[1]

fp = f"../data/{dataset}/PLM_data/item_emb_vicuna.npy"
embed = np.load(fp)

print(embed.shape)


sim_matrix = cosine_similarity(embed)
print("Similarity matrix computed.")

sorted_indice = np.argsort(-sim_matrix, axis=1)
print("Sorted.")

fp_indice = f"../data/{dataset}/PLM_data/item_indice.npy"
print(sorted_indice.shape, sorted_indice[0])
np.save(fp_indice, sorted_indice)
print("Saved.")
