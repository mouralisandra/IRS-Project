from modules.similarity import *
import time
import pandas as pd

# indexmap = getIndexMap()
indexmap= makeIndex()

def get_document_info(csv_file, doc_ids):
    if isinstance(doc_ids, dict):
        doc_ids = list(doc_ids.keys())
    # doc_ids=doc_ids[::-1]
    df = pd.read_csv(csv_file)
    # print(doc_ids)
    df['ID'] = df['ID'].astype(int)
    filtered_df = df[df['ID'].isin(doc_ids)]
    id_to_info = dict(zip(filtered_df['ID'], zip(filtered_df['Title'], filtered_df['Description'])))
    return id_to_info

def searchQuery(query):
    queryToken = preProcess(query)
    qweigth = calculate_document_term_weights(indexmap, queryToken)
    weightindexa = calculate_query_term_weights(qweigth, indexmap)
    print("Query Vector:", queryToken)
    print("Document Vector:", qweigth)
    print("Weight Vector:", weightindexa)
    sim = calculate_cosine_similarity(qweigth, weightindexa)
    sim = {k: sim[k] for k in reversed(sim.keys())}
    return sim


def searchAndRank(query):
    start_time = time.time()
    sim = searchQuery(query)
    end_time = time.time()

    n= 10

    top_10 = dict(list(sim.items())[:n])

    csv_file_path_main = 'data/mainn.csv'
    doc_info_dict = get_document_info(csv_file_path_main, top_10)

    results_table = "<table border='1'><tr><th>Rank</th><th>ID</th><th>Disease</th><th>Similarity Score</th><th>Similar Symptoms</th></tr>Execution time : <span class=green> " + str(
        round(
            end_time - start_time, 5)) + " sec</span"

    for rank, (doc_id, similarity_score) in enumerate(top_10.items(), start=0 + 1):
        doc_info = doc_info_dict.get(
            doc_id, ("N/A", "N/A"))
        doc_name, doc_description = doc_info

        results_table += f"<tr><td>{rank}</td><td>{doc_id}</td><td>{doc_name}</td><td>{similarity_score}</td><td>{doc_description}</td></tr>"

    results_table += "</table>"

    return results_table, top_10
