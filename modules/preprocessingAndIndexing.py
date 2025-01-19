from collections import defaultdict
import re
import pandas as pd
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
inverted_index = defaultdict(dict)
data = pd.read_csv('data/mainn.csv')
tokenindex = []
total_doc = len(data)

def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(word.strip() for word in file)
    return stop_words

def preProcess(pattern):
    w = tokenize_without_numbers(pattern)
    w2 = removal_of_stop_words(w)
    w3 = stemming(w2)
    # w3=[lemmatizer.lemmatize(word) for word in w3]
    return w3


def tokenize_without_numbers(text):
    text = text.strip()
    pattern = r'\b[a-zA-Z]+\b'
    tokens = re.findall(pattern, text)
    return [token.lower() for token in tokens]

def removal_of_stop_words(words):
    return [word for word in words if word not in stop_words]

stemming_rules = [
    ("sses", "ss"),
    ("ies", "i"),
    ("ss", "ss"),
    ("s", ""),
    ("tions", "t"),
    ("ative", ""),
    ("atives", ""),
    ("ize", ""),
    ("izes", "ize"),
    ("izing", "ize"),
    ("ized", "ize"),
    ("izing", "ize"),
    ("izing", "ize"),
    ("izing", "ize"),
    ("ized", "ize"),
    ("ish", ""),
    ("ism", ""),
    ("ist", ""),
    ("al", ""),
    ("ate", ""),
    ("en", ""),
    ("ify", ""),
    ("tive", ""),
    ("tives", ""),
    ("ic", ""),
    ("ics", ""),
    ("ical", ""),
    ("ically", ""),
    ("icity", ""),
    ("ionize", "ion"),
    ("ionizes", "ionize"),
    ("ionizing", "ionize"),
    ("ionized", "ionize"),
    ("ional", ""),
    ("ionally", ""),
    ("ioning", "ion"),
    ("ionings", "ioning"),
    ("ioned", "ion"),
    ("ioner", "ion"),
    ("ioners", "ioner"),
    ("ionable", "ion"),
    ("ionables", "ionable"),
    ("ioning", "ion"),
    ("ionings", "ioning"),
    ("ization", "ize"),
    ("izations", "ize"),
    ("izations", "ize"),
    ("izational", "ize"),
    ("izationally", "ize"),
    ("izationing", "ize"),
    ("izationings", "ize"),
    ("izations", "ize"),
    ("izationed", "ize"),
    ("izations", "ize"),
    ("ishly", ""),
    ("ishness", ""),
    ("ishnesses", "ishness"),
    ("ism", ""),
    ("ist", ""),
    ("istic", ""),
    ("istically", ""),
    ("istical", "ist"),
    ("istically", "ist"),
    ("istical", "ist"),
    ("istication", "ist"),
    ("istications", "istication"),
    ("isticated", "isticate"),
    ("isticate", "ist"),
    ("istically", "ist"),
    ("istical", "ist"),
    ("isticatedly", "isticate"),
    ("isticatedness", "isticate"),
    ("istication", "isticate"),
    ("istications", "isticate"),
    ("evening", "evening"),
    ("morning", "morning"),
    ("ism", ""),
    ("isms", "ism"),
    ("ist", ""),
    ("ist", ""),
    ("ists", "ist"),
    ("ist", ""),
    ("ist", ""),
    ("ists", "ist"),
    ("ist", ""),
    ("al", ""),
    ("ally", ""),
    ("al", ""),
    ("ally", ""),
    ("ed", ""),
    ("ing", ""),
    ("er", ""),
    ("or", ""),
    ("ar", ""),
    ("ary", ""),
    ("ery", ""),
    ("ful", ""),
    ("less", ""),
    ("ness", ""),
    ("ship", ""),
    ("sion", ""),
    ("tion", "t"),
    ("ive", ""),
    ("ize", ""),
    ("izing", "ize"),
    ("ized", "ize"),
    ("al", ""),
    ("ally", ""),
    ("ed", ""),
    ("ing", ""),
    ("er", ""),
    ("or", ""),
    ("ar", ""),
    ("ary", ""),
    ("ery", ""),
    ("ful", ""),
    ("less", ""),
    ("ness", ""),
    ("ship", ""),
    ("sion", ""),
    ("tion", "t"),
    ("ive", ""),
    ("ize", ""),
    ("izing", "ize"),
    ("ized", "ize"),
    ("ish", ""),
    ("ism", ""),
    ("ist", ""),
    ("al", ""),
    ("ate", ""),
    ("en", ""),
    ("ify", ""),
    ("ise", ""),
    ("ises", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ish", ""),
    ("ism", ""),
    ("ist", ""),
    ("al", ""),
    ("ate", ""),
    ("en", ""),
    ("ify", ""),
    ("ise", ""),
    ("ises", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ish", ""),
    ("ism", ""),
    ("ist", ""),
    ("al", ""),
    ("ate", ""),
    ("en", ""),
    ("ify", ""),
    ("ise", ""),
    ("ises", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ish", ""),
    ("ism", ""),
    ("ist", ""),
    ("al", ""),
    ("ate", ""),
    ("en", ""),
    ("ify", ""),
    ("ise", ""),
    ("ises", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ish", ""),
    ("ism", ""),
    ("ist", ""),
    ("al", ""),
    ("ate", ""),
    ("en", ""),
    ("ify", ""),
    ("ise", ""),
    ("ises", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ising", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ised", "ise"),
    ("ish", ""),
    ("ism", ""),
    ("ist", ""),
    ("al", ""),
    ("tive", ""),
    ("tives", ""),
    ("ic", ""),
    ("ics", ""),
    ("ical", ""),
    ("ically", ""),
    ("icity", ""),
    ("ionize", "ion"),
    ("ionizes", "ionize"),
    ("ionizing", "ionize"),
    ("ionized", "ionize"),
    ("ional", ""),
    ("ionally", ""),
    ("ioning", "ion"),
    ("ionings", "ioning"),
    ("ioned", "ion"),
    ("ioner", "ion"),
    ("ioners", "ioner"),
    ("ionable", "ion"),
    ("ionables", "ionable"),
    ("ioning", "ion"),
    ("ionings", "ioning"),
    ("ization", "ize"),
    ("izations", "ize"),
    ("izations", "ize"),
    ("izational", "ize"),
    ("izationally", "ize"),
    ("izationing", "ize"),
    ("izationings", "ize"),
    ("izations", "ize"),
    ("izations", "ize"),
    ("izations", "ize"),
    ("izationed", "ize"),
    ("izations", "ize"),
    ("izations", "ize"),
    ("izations", "ize"),
    ("ish", ""),
    ("ishly", ""),
    ("ishness", ""),
    ("ishnesses", "ishness"),
    ("ism", ""),
    ("ist", ""),
    ("istic", ""),
    ("istically", ""),
    ("istical", "ist"),
    ("istically", "ist"),
    ("istical", "ist"),
    ("istication", "ist"),
    ("istications", "istication"),
    ("isticated", "isticate"),
    ("isticate", "ist"),
    ("istically", "ist"),
    ("istical", "ist"),
    ("isticatedly", "isticate"),
    ("isticatedness", "isticate"),
    ("isticatednesses", "isticatedness"),
    ("istication", "isticate"),
    ("istications", "isticate"),
    ("ism", ""),
    ("isms", "ism"),
    ("ist", ""),
    ("ist", ""),
    ("ists", "ist"),
    ("ist", ""),
    ("ist", ""),
    ("ists", "ist"),
    ("ist", ""),
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
    ("al", ""),
    ("ance", ""),
    ("ence", ""),
    ("er", ""),
    ("ic", ""),
    ("able", ""),
    ("ible", ""),
    ("ant", ""),
    ("ement", ""),
    ("ment", ""),
    ("ent", ""),
    ("ou", ""),
    ("ism", ""),
    ("ate", ""),
    ("iti", ""),
    ("ous", ""),
    ("ive", ""),
    ("ize", ""),

]

def stemming(all_words):
    stemmed_word = []
    for w in all_words:
        for suffix, replace in stemming_rules:
            if w.endswith(suffix):
                w = w[:-len(suffix)] + replace
        stemmed_word.append(w)
    return stemmed_word

def count_string_occurrences(string, array_of_strings):
    return array_of_strings.count(string)

def makeIndex():
    indexmap = defaultdict(dict)  
    terms = set()  
    
    for doc_id, row in data.iterrows():
        document = str(row['Description'])
        title_terms = preProcess(str(row['Title']))
        predoc = preProcess(document)
        tokenindex.append(predoc)
        terms.update(predoc + title_terms)
        for term in set(predoc + title_terms):
            tf = count_string_occurrences(term, predoc + title_terms)
            indexmap[term][doc_id+2] = tf

    with open("index.txt", "w") as file:
        for term, doc_dict in indexmap.items():
            file.write(f"{term} -> {', '.join(f'({doc_id}, {tf})' for doc_id, tf in doc_dict.items())}\n")

    return indexmap


def getIndexMap():
    indexmap = defaultdict(dict)

    with open("indexmovies.txt", "r") as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                # Split the term and its document mappings
                term, doc_data = line.split(" -> ")
                term = term.strip()
                
                # Extract document ID and term frequency (tf) pairs
                doc_pairs = re.findall(r'\((\d+), (\d+)\)', doc_data)
                for doc_id, tf in doc_pairs:
                    indexmap[term][int(doc_id)] = int(tf)
            except ValueError:
                print(f"Skipping malformed line: {line}")

    return indexmap



