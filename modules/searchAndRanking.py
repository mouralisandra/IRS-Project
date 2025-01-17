from modules.similarity import *
import time
import pandas as pd
csv_file_path_main = 'data/mainn.csv'
docs = pd.read_csv(csv_file_path_main)
total_doc=len(docs)
# indexmap = getIndexMap()
indexmap= makeIndex()

def get_document_info(csv_file, doc_ids):
    if isinstance(doc_ids, dict):
        doc_ids = list(doc_ids.keys())
    
    df = pd.read_csv(csv_file, usecols=['ID', 'Title', 'Description'])
    df['ID'] = df['ID'].astype(int)
    filtered_df = df[df['ID'].isin(doc_ids)]
    id_to_info = dict(zip(filtered_df['ID'], zip(filtered_df['Title'], filtered_df['Description'])))
    return id_to_info


def searchQuery(query, tf_choice_q='l', idf_choice_q='t', tf_choice_d='l', idf_choice_d='t', norm_choice_q='n', norm_choice_d='n'):
    queryToken = preProcess(query)
    tokens = [token for token in queryToken if token in indexmap]
    document_weights, normalized_query_weights, sim = calculate_tfidf_similarity(indexmap, tokens, total_doc, tf_choice_q, idf_choice_q, tf_choice_d, idf_choice_d, norm_choice_q, norm_choice_d)
    print(sim)
    sim = {k: sim[k] for k in sorted(sim.keys(), key=lambda x: sim[x], reverse=True)}
    return sim


def searchAndRank(query, n=10, tf_choice_q='l', idf_choice_q='t', tf_choice_d='l', idf_choice_d='t', norm_choice_q='n', norm_choice_d='n'):
    start_time = time.time()
    sim = searchQuery(query, tf_choice_q, idf_choice_q, tf_choice_d, idf_choice_d, norm_choice_q, norm_choice_d)
    end_time = time.time()

    top_10 = dict(list(sim.items()))
    doc_info_dict = get_document_info(csv_file_path_main, top_10)

    results_table = "<table border='1'><tr><th>Rank</th><th>ID</th><th>Disease</th><th>Similarity Score</th><th>Similar Symptoms</th></tr>Execution time : <span class=green> " + str(
        round(end_time - start_time, 5)) + " sec</span>"

    for rank, (doc_id, similarity_score) in enumerate(top_10.items(), start=0+ 1):
        doc_info = doc_info_dict.get(doc_id, ("N/A", "N/A"))
        doc_name, doc_description = doc_info

        results_table += f"<tr><td>{rank}</td><td>{doc_id}</td><td>{doc_name}</td><td>{similarity_score}</td><td>{doc_description}</td></tr>"

    results_table += "</table>"

    return results_table, top_10
