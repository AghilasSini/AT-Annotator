
# Baseline
import sys
import codecs
import logging
import os
import re
from collections import defaultdict
from lxml import etree
from collections import OrderedDict


from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn import tree

# now serious example

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


#
from sklearn.svm import LinearSVC


import pandas as pd

# Argument
import argparse
import numpy  as np

# Debug
import traceback
import time

# text processing and regular expression
import re
import string




import spacy
from spacy.pipeline import Tagger
from spacy import displacy
from collections import defaultdict

__module_dir = os.path.dirname(__file__)

	# Preprocessing and tokenizing
def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

class SentenceParser():
	def __init__(self,model="fr_core_news_md"):
		self.nlp=spacy.load(model)
		self.tagger = Tagger(self.nlp.vocab)
		self.parsed_text=OrderedDict()

	def parsing_sentence(self,sentence,dictionary):
		doc=self.nlp(sentence)
		sentence_parsing_info=list()
		for sent in doc.sents:
			for token in sent:
				(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)








def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	parser.add_argument("path", help="The input file to be projected")

	# Parsing arguments
	args = parser.parse_args()

	dataset=pd.read_csv(args.path,sep='\t')

	data=dataset['text'].to_list()
	target_data=dataset['label'].to_list()
	# sent_parser=SentenceParser()
	# for idx,(sent, label) in enumerate(zip(data,target_data)):
	# 	# sent_parser.parsing_sentence(sent)
	# 	sampleFilename=os.path.join('./dataset/input/','{}_{}.txt'.format(label.lower(),str(idx).zfill(5)))
	# 	with codecs.open(sampleFilename,'w','utf-8') as fl:
	# 		fl.write("{}\n".format(sent))


	X_train, X_test, y_train, y_test = train_test_split(data, target_data, test_size=0.3)



	text_clf = Pipeline([('vect', CountVectorizer()),
	                     ('tfidf', TfidfTransformer()),
	                     # ('clf', NearestCentroid()),
	                     # ('clf', RandomForestClassifier(n_estimators=100)),
	                     # ('clf', LinearSVC()),
	                     # ('clf', MultinomialNB()),
						 # ('clf', KNeighborsClassifier()),
						('clf', tree.DecisionTreeClassifier()),
							  # ('clf', BaggingClassifier(KNeighborsClassifier())),
							# ('clf', GradientBoostingClassifier(n_estimators=100)),


	                     ])


	text_clf.fit(X_train, y_train)

	predicted = text_clf.predict(X_test)

	print(metrics.classification_report(y_test, predicted))












if __name__ == '__main__':
	main()