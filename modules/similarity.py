import numpy as np
import math
import random
from modules.preprocessingAndIndexing import *


def calculate_document_term_weights(indexmap, token_list):
    """
    Calculate TF-IDF weights for terms in a document or query.
    This is used to represent the importance of terms in a document or query.

    Args:
        indexmap (dict): Inverted index mapping terms to documents and their frequencies.
        token_list (list): List of tokens (terms) from a document or query.

    Returns:
        dict: A dictionary where each term is mapped to its TF-IDF weight.
    """
    term_weights = {}
    for token in token_list:
        if token in term_weights:
            continue  # Skip if the term's weight is already calculated
        else:
            # Term Frequency (TF): Logarithmic scaling of term frequency in the document/query
            tf = 1 + math.log(count_string_occurrences(token, token_list), 10)
            # Inverse Document Frequency (IDF): Logarithmic scaling of how rare the term is across documents
            idf = math.log(total_doc / (len(indexmap.get(token, {})) +0.1), 10) 
            # TF-IDF weight: Combines TF and IDF to represent term importance
            term_weights[token] = tf * idf

    return term_weights


def calculate_query_term_weights(query_weights, indexmap):
    """
    Calculate TF-IDF weights for terms in the inverted index based on the query.
    This is used to represent the importance of terms in documents relative to the query.

    Args:
        query_weights (dict): TF-IDF weights for the query terms.
        indexmap (dict): Inverted index mapping terms to documents and their frequencies.

    Returns:
        dict: A dictionary where each document is mapped to its terms and their TF-IDF weights.
    """
    document_weights = {}
    for token in query_weights:
        if token in indexmap:
            # IDF for the term: Logarithmic scaling of how rare the term is across documents
            idf = math.log(total_doc / (len(indexmap.get(token, {})) +0.1), 10)
            for doc_id in indexmap[token]:
                # TF for the term in the document: Logarithmic scaling of term frequency
                tf = 1 + math.log(indexmap[token][doc_id], 10)
                # TF-IDF weight for the term in the document
                document_weights.setdefault(doc_id, {})
                document_weights[doc_id][token] = tf * idf 

    return document_weights


def calculate_vector_magnitude(vector):
    """
    Calculate the magnitude (Euclidean norm) of a vector.
    This is used in cosine similarity calculations to normalize vectors.

    Args:
        vector (dict): A dictionary representing a vector (e.g., TF-IDF weights).

    Returns:
        float: The magnitude of the vector.
    """
    return np.linalg.norm(list(vector.values()))


def calculate_cosine_similarity(query_weights, document_weights):
    """
    Calculate cosine similarity between a query and all documents.
    This is used to rank documents based on their relevance to the query.

    Args:
        query_weights (dict): TF-IDF weights for the query terms.
        document_weights (dict): TF-IDF weights for all documents.

    Returns:
        dict: A dictionary where each document is mapped to its cosine similarity score with the query.
    """
    similarity_scores = {}
    query_magnitude = calculate_vector_magnitude(query_weights)

    for doc_id, doc_vector in document_weights.items():
        doc_magnitude = calculate_vector_magnitude(doc_vector)

        if doc_magnitude == 0 or query_magnitude == 0:
            # Avoid division by zero; set similarity to 0 if either vector has zero magnitude
            similarity_scores[doc_id] = 0
        else:
            # Find common terms between the query and the document
            common_terms = set(query_weights.keys()) & set(doc_vector.keys())
            # Convert query and document vectors to numpy arrays for common terms
            query_vector = np.array([query_weights[term] for term in common_terms])
            doc_vector = np.array([doc_vector[term] for term in common_terms])
            # Cosine similarity: Dot product of vectors divided by the product of their magnitudes
            similarity_scores[doc_id] = np.dot(doc_vector, query_vector) / (doc_magnitude *query_magnitude) 
    

    # Sort documents by similarity scores in descending order
    return dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True))
