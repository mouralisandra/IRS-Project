import math
import numpy as np

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
