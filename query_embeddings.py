import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained SBERT model
#model = SentenceTransformer('all-MiniLM-L6-v2')
model =SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def generate_embedding(query):
    """Generates SBERT embeddings for a given query."""
    return model.encode(query, convert_to_numpy=True)


def identify_related_queries(session_queries, current_query, threshold=0.7):
    """
    Identifies related queries from a session using cosine similarity.

    Parameters:
        session_queries (list): List of previous queries in the session.
        current_query (str): The latest query entered by the user.
        threshold (float): Similarity threshold to consider queries as related.

    Returns:
        list: Related queries from the session.
    """
    # Generate embeddings for session queries and current query
    session_embeddings = np.array([generate_embedding(q) for q in session_queries])
    current_embedding = generate_embedding(current_query).reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(current_embedding, session_embeddings)[0]

    # Retrieve related queries based on threshold
    related_queries = [session_queries[i] for i in range(len(session_queries)) if similarities[i] >= threshold]

    return related_queries


def compute_average_vector(queries):
    """
    Computes the average vector embedding for a set of related queries.

    Parameters:
        queries (list): List of related queries.

    Returns:
        np.array: Averaged vector representation.
    """
    if not queries:
        return None

    query_embeddings = np.array([generate_embedding(q) for q in queries])
    avg_vector = np.mean(query_embeddings, axis=0)

    return avg_vector


# Example session queries
session_queries = [
    "What is natural language processing?",
    "Explain machine learning in NLP",
    "How does BERT work for text classification?"
]

# Current user query
current_query = "How does NLP work in text analysis?"

# Identify related queries
related_queries = identify_related_queries(session_queries, current_query)

# Compute average vector for related queries
average_vector = compute_average_vector(related_queries)

# Print results
print("Current Query:", current_query)
print("Related Queries:", related_queries)
print("Average Vector Shape:", average_vector.shape if average_vector is not None else "No related queries found")
