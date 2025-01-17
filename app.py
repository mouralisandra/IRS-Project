from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from modules.searchAndRanking import searchAndRank
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

def correct_spelling(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_user_input', methods=['POST'])
def process_user_input():
    data = request.json
    query = data.get("query", "")
    n = data.get("n", 10)  # Default is 10
    tf_choice_q = data.get('tf_choice_q')
    idf_choice_q = data.get('idf_choice_q')
    tf_choice_d = data.get('tf_choice_d')
    idf_choice_d = data.get('idf_choice_d')
    norm_choice_q = data.get('norm_choice_q')
    norm_choice_d = data.get('norm_choice_d')
    print(tf_choice_q, idf_choice_q, tf_choice_d, idf_choice_d, norm_choice_q, norm_choice_d)
    # corrected_query = correct_spelling(query)
    did_you_mean = None
    top_n=None
    # if corrected_query != query:
    #     did_you_mean = {
    #         "suggestion": corrected_query,
    #         "original": query
    #     }
    #     search_results_table, top_n = searchAndRank(corrected_query, n, tf_choice_q, idf_choice_q, tf_choice_d, idf_choice_d, norm_choice_q, norm_choice_d)
    # else:
    search_results_table, top_n = searchAndRank(query, n, tf_choice_q, idf_choice_q, tf_choice_d, idf_choice_d, norm_choice_q, norm_choice_d)
    print(top_n)
    return jsonify({
        "results": search_results_table,
        "did_you_mean": did_you_mean
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
