import nltk

from gensim.models import Doc2Vec
import numpy as np
from numpy import dot
from gensim import utils, matutils
import pandas as pd

import argparse




def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	
	parser.add_argument('doc2vec_model_dbow',help='dbow model synpaflex corpus')
	parser.add_argument("sentiment_analysis", help="The input file to be projected")

	
	args = parser.parse_args()
	print('load doc2vec model dbow')
	# f = open(args.doc2vec_model_dbow, "r+b")
	model_dbow =  Doc2Vec.load(args.doc2vec_model_dbow)
	# m = g.Doc2Vec.load(model)
	# f.close()
	print('loading process is finished with success')
	print('load data sentiment data set')
	df_=pd.read_csv(args.sentiment_analysis,sep='|')
	print('loading sentiment data is finished with success')
	print(df_.head())



	

if __name__ == '__main__':
	main()
