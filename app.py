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
    page_no = 1

    corrected_query = correct_spelling(query)
    did_you_mean = None

    if corrected_query != query:
        did_you_mean = {
            "suggestion": corrected_query,
            "original": query
        }
        search_results_table, _ = searchAndRank(corrected_query, page_no)
    else:
        search_results_table, _ = searchAndRank(query, page_no)

    return jsonify({
        "results": search_results_table,
        "did_you_mean": did_you_mean
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
