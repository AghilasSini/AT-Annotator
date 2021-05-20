import numpy as np
import re
import string


import pandas as pd

corpus = """
Simple example with Cats and Mouse
Another simple example with dogs and cats
Another simple example with mouse and cheese
""".split("\n")[1:-1]




# clearing and tokenizing
l_A = corpus[0].lower().split()
l_B = corpus[1].lower().split()
l_C = corpus[2].lower().split()



# Calculating bag of words
word_set = set(l_A).union(set(l_B)).union(set(l_C))

word_dict_A = dict.fromkeys(word_set, 0)
word_dict_B = dict.fromkeys(word_set, 0)
word_dict_C = dict.fromkeys(word_set, 0)

for word in l_A:
	word_dict_A[word] += 1

for word in l_B:
	word_dict_B[word] += 1

for word in l_C:
	word_dict_C[word] += 1

print('bag of word')

print(word_dict_C)

# compute term frequency

def compute_tf(word_dict, l):
	tf = {}
	sum_nk = len(l)
	for word, count in word_dict.items():
		tf[word] = count/sum_nk
	return tf
  
  
tf_A = compute_tf(word_dict_A, l_A)
tf_B = compute_tf(word_dict_B, l_B)
tf_C = compute_tf(word_dict_C, l_C)



# compute idf

def compute_idf(strings_list):
    n = len(strings_list)
    idf = dict.fromkeys(strings_list[0].keys(), 0)
    for l in strings_list:
        for word, count in l.items():
            if count > 0:
                idf[word] += 1
    
    for word, v in idf.items():
        idf[word] = np.log(n / float(v))
    return idf


    
idf = compute_idf([word_dict_A, word_dict_B, word_dict_C])



def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf
    
tf_idf_A = compute_tf_idf(tf_A, idf)
tf_idf_B = compute_tf_idf(tf_B, idf)
tf_idf_C = compute_tf_idf(tf_C, idf)


print('with tf idf')
tf_idf_simple=pd.DataFrame([tf_idf_A, tf_idf_B, tf_idf_C])
print(tf_idf_simple.head())


# now serious example

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

all_text  =  """"
Google and Facebook are strangling the free press to death. Democracy is the loserGoogle an
Your 60-second guide to security stuff Google touted today at Next '18
A Guide to Using Android Without Selling Your Soul to Google
Review: Lenovo’s Google Smart Display is pretty and intelligent
Google Maps user spots mysterious object submerged off the coast of Greece - and no-one knows what it is
Android is better than IOS
In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency
is a numerical statistic that is intended to reflect
how important a word is to a document in a collection or corpus.
It is often used as a weighting factor in searches of information retrieval
text mining, and user modeling. The tf-idf value increases proportionally
to the number of times a word appears in the document
and is offset by the frequency of the word in the corpus
""".split("\n")[1:-1]

# Preprocessing and tokenizing
def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing)
tfidf = tfidf_vectorizer.fit_transform(all_text)



kmeans = KMeans(n_clusters=2).fit(tfidf)


lines_for_predicting = ["tf and idf is awesome!"]
print(tfidf_vectorizer.transform(corpus))
# kmeans.predict(tfidf_vectorizer.transform(lines_for_predicting))
