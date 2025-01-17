import numpy as np
from modules.preprocessingAndIndexing import *
from modules.tfidf import TF_FORMULAS, IDF_FORMULAS, NORM_FORMULAS

def calculate_tfidf_similarity(indexmap, tokens, total_doc, tf_choice_q='l', idf_choice_q='t', tf_choice_d='l', idf_choice_d='t', norm_choice_q='n', norm_choice_d='n'):
    tf_formula_q = TF_FORMULAS[tf_choice_q]
    idf_formula_q = IDF_FORMULAS[idf_choice_q]
    tf_formula_d = TF_FORMULAS[tf_choice_d]
    idf_formula_d = IDF_FORMULAS[idf_choice_d]
    common_tokens = set(tokens)
    
    # Step 1: Document TF-IDF calculation
    document_weights = {}
    for doc in range(total_doc):
        if doc not in document_weights:
            document_weights[doc] = {}  
        for token in common_tokens:
            if token not in document_weights[doc]:
                document_weights[doc][token] = 0
    for token in common_tokens:
        if token in indexmap:
            idf_d = idf_formula_d(total_doc, len(indexmap.get(token)))
            print(token, idf_d,indexmap[token])
            for doc_id in indexmap[token]:
                tf_d = tf_formula_d(indexmap[token][doc_id])
                print(tf_d*idf_d)
                if doc_id not in document_weights:
                    document_weights[doc_id] = {}
                document_weights[doc_id][token] = tf_d * idf_d 
    # keep document weights with at least a token value >0
    document_weights = {doc_id: doc_vector for doc_id, doc_vector in document_weights.items() if any(doc_vector.values())}

    # Step 2: Normalize document vectors
    for doc_id, doc_vector in document_weights.items():
        vector = np.array(list(doc_vector.values()))
        normalized_vector = NORM_FORMULAS[norm_choice_d](vector)
        normalized_weights = {token: normalized_vector[i] for i, token in enumerate(doc_vector)}
        document_weights[doc_id] = normalized_weights
    # print("document weights")
    # print(document_weights)

    # Step 3: Query TF-IDF calculation
    query_weights = {}
    for token in tokens:
        if token in common_tokens:
            tf_q = tf_formula_q(tokens.count(token))  
            idf_q = idf_formula_q(total_doc, len(indexmap.get(token, {})))
            query_weights[token] = tf_q * idf_q

    # Step 4: Normalize query vector
    query_vector = np.array(list(query_weights.values()))
    for i, token in enumerate(query_weights):
        query_vector[i] = query_vector[i] * idf_formula_q(total_doc, len(indexmap.get(token, {})))
   
    normalized_query_vector = NORM_FORMULAS[norm_choice_q](query_vector)
    normalized_query_weights = {token: normalized_query_vector[i] for i, token in enumerate(query_weights)}
    # print("query vector")
    # print(normalized_query_vector)

    # Step 5: Calculate similarity
    return cosine_similarity(normalized_query_vector, document_weights,normalized_query_weights)
    
def cosine_similarity(normalized_query_vector, document_weights,normalized_query_weights):
    similarities = {}
    query_vector_norm = np.linalg.norm(normalized_query_vector)
    
    for doc_id, doc_vector in document_weights.items():
        doc_vector_values = np.array(list(doc_vector.values()))
        doc_vector_norm = np.linalg.norm(doc_vector_values)

        if doc_vector_values.shape != normalized_query_vector.shape:
            doc_vector_values = np.resize(doc_vector_values, normalized_query_vector.shape)

        dot_product = np.dot(doc_vector_values, normalized_query_vector)
        cosine_similarity = dot_product / (doc_vector_norm * query_vector_norm) if doc_vector_norm and query_vector_norm else 0
        similarities[doc_id] = cosine_similarity

    return document_weights, normalized_query_weights, similarities