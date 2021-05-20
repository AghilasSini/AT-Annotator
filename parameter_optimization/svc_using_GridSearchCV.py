
# Baseline
import sys
import codecs
import logging
import os
import re
from collections import defaultdict
from lxml import etree
from collections import OrderedDict

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC




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
from sklearn.metrics import classification_report, confusion_matrix 


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


def clean_text(Text):
	Text = re.sub("\n", " ", Text)
	Text = re.sub("--", "", Text)
	Text = re.sub("\.\.\.", ".", Text)
	Text = Text.lower()
	# Text = re.split("[.!?]", Text)
	# Sent = re.split("\W", Text)
	# Sent = [Token for Token in Sent if Token]
	return Text

# convert text to TF-IDF:

def TFIDF(X_train,X_test,MAX_NB_WORDS=100):
		vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
		X_train = vectorizer_x.fit_transform(X_train).toarray()
		X_test = vectorizer_x.transform(X_test).toarray()
		print("tf-idf with",str(np.array(X_train).shape[1]),"features")
		return (X_train,X_test)

# 'roc_plot_CNN-SVM_classifier.png'
def roc_curve(y_true,y_pred,outputFilename):
	import numpy as np
	import matplotlib.pyplot as plt
	from itertools import cycle


	from sklearn.metrics import roc_curve, auc
	from sklearn.preprocessing import label_binarize
	from sklearn.multiclass import OneVsRestClassifier
	from scipy import interp
	from sklearn.metrics import roc_auc_score
	y_test = label_binarize(y_true, classes=[0, 1, 2])
	y_score = label_binarize(y_pred, classes=[0, 1, 2])
	n_classes = y_test.shape[1]

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


	##############################################################################
	# Plot of a ROC curve for a specific class
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic of CNN-SVM classifier')
	plt.legend(loc="lower right")
	plt.show()


	##############################################################################
	# Plot ROC curves for the multilabel problem
	# ..........................................
	# Compute macro-average ROC curve and ROC area

	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# Plot all ROC curves
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]),
	         color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr["macro"], tpr["macro"],
	         label='macro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["macro"]),
	         color='navy', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic of SVM classifier')
	plt.legend(loc="lower right")
	plt.savefig(outputFilename)



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

	convert_dict={
	'label':'category',
			
	}
	# # print(cat_list)
	dataset = dataset.astype(convert_dict)
	dataset['label_cat'] = dataset.label.cat.codes


	data=[]
	for frame in dataset['text'].to_list():
		data.append(clean_text(frame))

	target_data=dataset['label_cat'].to_list()
	

	x_train, x_test, y_train, y_test = train_test_split(data, target_data, test_size=0.3, random_state=42)



	x_train_tfidf,x_test_tfidf=TFIDF(x_train,x_test)
	# defining parameter range 
	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','linear']
              }  
  
	# for cv in [5,6,7,8,9,10]:
	grid = GridSearchCV(SVC(), param_grid, refit = True,cv=10) 
	grid.fit(x_train_tfidf, y_train)
	print(grid.best_params_) 
	print(grid.best_estimator_) 
	grid_predictions = grid.predict(x_test_tfidf) 
	print(classification_report(y_test, grid_predictions)) 
	print("Accuracy for SVM on CV data: ",accuracy_score(y_test,grid_predictions))
	roc_curve(y_test,grid_predictions,'plot_roc_SVM.png')


if __name__ == '__main__':
	main()


