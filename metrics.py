import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cosine
from statistics import mean 
from tqdm import tqdm


def calculate_hit_rate_at_k(test_dir, dam_dir, k):
    """Functions that calculates thet hit rate at k.

    Args:
        test_dir (string): path of the directory containing test image embeddings
        dam_dir (string): path of the directory containing DAM image embeddings
        k (int): the number of the retrived images to be considered for the evaluation

    Returns:
        float: the score of the evaluation
    """
    
    column_names = ['test_name', 'ref_name']
    annotations = pd.read_csv('annotations.csv', header=None, names=column_names)

    annotations['test_name'] = annotations['test_name'].apply(remove_extension)
    annotations['ref_name'] = annotations['ref_name'].apply(remove_extension)
    
    hit_rate_at_k_list = []

    for test_emb_name in tqdm(os.listdir(test_dir), desc="Evaluating on test set"):
        
        test_emb = np.load(os.path.join(test_dir, test_emb_name))  # Load test embedding

        cosine_similarities = []

        for dam_emb_name in os.listdir(dam_dir):
            dam_emb = np.load(os.path.join(dam_dir, dam_emb_name))  # Load dam embedding
            similarity = 1 - cosine(test_emb, dam_emb) 
            cosine_similarities.append((os.path.splitext(dam_emb_name)[0], similarity))

        cosine_similarities.sort(key=lambda x: x[1], reverse=True)
        ref_name = annotations.loc[annotations['test_name'] == os.path.splitext(test_emb_name)[0], 'ref_name'].values[0]

        # Find the rank of the reference image
        rank = next((i for i, (dam_name, _) in enumerate(cosine_similarities) if dam_name == ref_name), None)
        
        hit_rate_at_k_list.append(hit_rate_at_k(rank, k))
       
    return mean(hit_rate_at_k_list)


def precision_at_1(rank):
    return 1 if rank==1 else 0
    
def hit_rate_at_k(rank, k=3):
    return 1 if rank<=k else 0

def mrr(rank):
    return 1 / rank

def remove_extension(filename):
    return os.path.splitext(filename)[0]
