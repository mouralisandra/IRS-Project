import numpy as np
import math
from modules.preprocessingAndIndexing import *

# TF Formula Definitions:
def tf_n(count):
    return count
def tf_l(count):
    return 1 + math.log(count) if count > 0 else 0

# IDF Formula Definitions:
def idf_n(N, df):
    return 1

def idf_t(N, df):
    return math.log(N / df) if df > 0 else 0

def idf_p(N, df):
    return math.log((N - df) / df) if df > 0 and df < N else 0

# Normalization Functions:
def norm_n(vector):
    return vector

def norm_c(vector):
    """ Cosine normalization: normalize the vector using its magnitude """
    norm_val = np.linalg.norm(vector)
    return vector / norm_val if norm_val > 0 else vector

TF_FORMULAS = {
    'n': tf_n,
    'l': tf_l
}

IDF_FORMULAS = {
    'n': idf_n,
    't': idf_t,
    'p': idf_p
}

NORM_FORMULAS = {
    'n': norm_n,
    'c': norm_c
}


def calculate_document_term_weights(indexmap, token_list, tf_choice='l', idf_choice='t', norm_choice='n'):
    """
    Calculate TF-IDF weights for terms in a document or query.
    This is used to represent the importance of terms in a document or query.
    """
    term_weights = {}
    tf_formula = TF_FORMULAS[tf_choice]
    idf_formula = IDF_FORMULAS[idf_choice]
    
    for token in token_list:
        if token in term_weights:
            continue  
        else:
            tf = tf_formula(count_string_occurrences(token, token_list))
            idf = idf_formula(total_doc, len(indexmap.get(token, {})))
            term_weights[token] = tf * idf
    vector = np.array(list(term_weights.values()))
    normalized_vector = NORM_FORMULAS[norm_choice](vector)
    normalized_weights = {token: normalized_vector[i] for i, token in enumerate(term_weights)}

    return normalized_weights


def calculate_query_term_weights(query_weights, indexmap, tf_choice='l', idf_choice='t', norm_choice='n'):
    """
    Calculate TF-IDF weights for terms in the inverted index based on the query.
    This is used to represent the importance of terms in documents relative to the query.
    """
    document_weights = {}
    tf_formula = TF_FORMULAS[tf_choice]
    idf_formula = IDF_FORMULAS[idf_choice]

    for token in query_weights:
        if token in indexmap:
            # IDF for the term
            idf = idf_formula(total_doc, len(indexmap.get(token, {})))
            for doc_id in indexmap[token]:
                # TF for the term in the document
                tf = tf_formula(indexmap[token][doc_id])
                # TF-IDF weight for the term in the document
                document_weights.setdefault(doc_id, {})
                document_weights[doc_id][token] = tf * idf 

    return document_weights


def calculate_cosine_similarity(query_weights, document_weights):
    """
    Calculate cosine similarity between a query and all documents.
    This is used to rank documents based on their relevance to the query.
    Returns:
        dict: A dictionary where each document is mapped to its cosine similarity score with the query.
    """
    similarity_scores = {}
    query_magnitude = np.linalg.norm(list(query_weights.values()))

    for doc_id, doc_vector in document_weights.items():
        doc_magnitude = np.linalg.norm(list(doc_vector.values()))

        if doc_magnitude == 0 or query_magnitude == 0:
            similarity_scores[doc_id] = 0
        else:
            common_terms = set(query_weights.keys()) & set(doc_vector.keys())
            query_vector = np.array([query_weights[term] for term in common_terms])
            doc_vector = np.array([doc_vector[term] for term in common_terms])
            similarity_scores[doc_id] = np.dot(doc_vector, query_vector) / (doc_magnitude * query_magnitude)
    
    return dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True))
