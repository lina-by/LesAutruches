from sklearn.metrics.pairwise import cosine_similarity

def find_top_k_matches(test_embedding, reference_embeddings, k=10):
    similarities = []
    
    for ref_name, ref_embedding in reference_embeddings.items():
        similarity = cosine_similarity([test_embedding], [ref_embedding])[0][0]
        similarities.append((ref_name, similarity))
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top k matches
    return similarities[:k]