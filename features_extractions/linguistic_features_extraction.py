
import os
import sys
import glob
import re
import nltk
import string
# nltk.download('punkt')
# nltk.download('stopwords') 
import codecs 
import matplotlib.pyplot as plt
from collections import defaultdict

import pandas as pd

from sklearn.model_selection import train_test_split
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import argparse
stemmer = FrenchStemmer()

import spacy
from spacy.pipeline import Tagger
from spacy import displacy
import sys,os
import codecs
from collections import OrderedDict




class SentenceParser():
	def __init__(self,model="fr_core_news_md"):
		self.nlp=spacy.load(model)
		self.tagger = Tagger(self.nlp.vocab)
		self.parsed_text=OrderedDict()

	def parsing_sentence(self,utt_text,utt_id,label):
		doc=self.nlp(utt_text)
		for sent in doc.sents:
			for token in sent:
				if not 'utterance' in self.parsed_text.keys():
					self.parsed_text['utterance']=[utt_id]
				else:
					self.parsed_text['utterance'].append(utt_id)

				if not 'token' in self.parsed_text.keys():
					self.parsed_text['token']=[token.text]
				else:
					self.parsed_text['token'].append(token.text)

				if not 'lemma' in self.parsed_text.keys():
					self.parsed_text['lemma']=[token.lemma_]
				else:
					self.parsed_text['lemma'].append(token.lemma_)

				if not 'stem' in self.parsed_text.keys():
					self.parsed_text['stem']=[stemmer.stem(token.text)]
				else:
					self.parsed_text['stem'].append(stemmer.stem(token.text))


				if not 'pos' in self.parsed_text.keys():
					self.parsed_text['pos']=[token.pos_]
				else:
					self.parsed_text['pos'].append(token.pos_)


				if not 'tag' in self.parsed_text.keys():
					self.parsed_text['tag']=[token.tag_]
				else:
					self.parsed_text['tag'].append(token.tag_)


				if not 'shape' in self.parsed_text.keys():
					self.parsed_text['shape']=[token.shape_]
				else:
					self.parsed_text['shape'].append(token.shape_)
				
				if not 'is_alpha' in self.parsed_text.keys():
					self.parsed_text['is_alpha']=[token.is_alpha]
				else:
					self.parsed_text['is_alpha'].append(token.is_alpha)
				if not 'label' in self.parsed_text.keys():
					self.parsed_text['label']=[label]
				else:
					self.parsed_text['label'].append(label)

	def pprint(self,outFilename):
		#convert to 
		df=pd.DataFrame.from_dict(self.parsed_text)
		df.to_csv(outFilename,index=False,sep='\t')



def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	parser.add_argument("input_dir", help="The input file to be projected")

	parser.add_argument("file_id_list", help="The input file to be projected")


	# Parsing arguments
	args = parser.parse_args()
	sentparser=SentenceParser()
	filelist=[]
	with codecs.open(args.file_id_list,'r','utf-8') as fl:
		filelist+=[os.path.join(args.input_dir,line.strip()+'.txt') for line in fl.readlines() ]

	
	for txtFn in filelist: 
		if os.path.exists(txtFn):
			uttFilename=os.path.basename(txtFn).split('.')[0]
			label,iutt=uttFilename.split('_')
			outFilename=os.path.splitext(txtFn)[0]+'.csv'
			print('parsing {}'.format(uttFilename))
			with codecs.open(txtFn,'r','utf-8') as  infile:
				sentparser.parsing_sentence(infile.readline().strip(),uttFilename,label)
				sentparser.pprint(outFilename)
			print('done')

if __name__ == '__main__':
	main()