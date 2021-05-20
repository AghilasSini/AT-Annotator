from sklearn.datasets import fetch_20newsgroups
from keras.layers import  Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split
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

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# convert text to TF-IDF:

def TFIDF(X_train,X_eval,X_test,MAX_NB_WORDS=75000):
		vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
		X_train = vectorizer_x.fit_transform(X_train).toarray()
		X_eval = vectorizer_x.transform(X_eval).toarray()
		X_test = vectorizer_x.transform(X_test).toarray()
		print("tf-idf with",str(np.array(X_train).shape[1]),"features")
		return (X_train,X_eval,X_test)


# Build a DNN Model for Text:
def Build_Model_DNN_Text(shape, nClasses, dropout=0.7):
		"""
		buildModel_DNN_Tex(shape, nClasses,dropout)
		Build Deep neural networks Model for text classification
		Shape is input feature space
		nClasses is number of classes
		"""
		model = Sequential()
		node = 512 # number of nodes
		nLayers = 3 # number of  hidden layer
		model.add(Dense(node,input_dim=shape,activation='relu'))
		model.add(Dropout(dropout))
		for i in range(0,nLayers):
			model.add(Dense(node,input_dim=node,activation='relu'))
			model.add(Dropout(dropout))
		model.add(Dense(nClasses, activation='softmax'))


		model.compile(loss='sparse_categorical_crossentropy',
									optimizer='adam',
									metrics=['accuracy'])

		return model



def clean_text(Text):
	Text = re.sub("\n", " ", Text)
	Text = re.sub("--", "", Text)
	Text = re.sub("\.\.\.", ".", Text)
	Text = Text.lower()
	# Text = re.split("[.!?]", Text)
	# Sent = re.split("\W", Text)
	# Sent = [Token for Token in Sent if Token]
	return Text







def main():
	# newsgroups_train = fetch_20newsgroups(subset='train')
	# newsgroups_test = fetch_20newsgroups(subset='test')
	# X_train = newsgroups_train.data
	# X_test = newsgroups_test.data
	# y_train = newsgroups_train.target
	# y_test = newsgroups_test.target
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
	
	X, X_test, y, y_test = train_test_split(data, target_data, test_size=0.3)
	X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.3)


	X_train_tfidf,X_eval_tfidf,X_test_tfidf = TFIDF(X_train,X_eval,X_test)

	# print(y_train)
	model_DNN = Build_Model_DNN_Text(X_train_tfidf.shape[1], 3)
	print(model_DNN.summary())
	model_DNN.fit(X_train_tfidf, y_train,validation_data=(X_eval_tfidf, y_eval),
															epochs=10,
															batch_size=16,
															verbose=2)

	predicted = model_DNN.predict(X_test_tfidf)
	predicted = np.argmax(predicted, axis=1)





	y_test = label_binarize(y_test, classes=[0, 1, 2])
	y_score = label_binarize(predicted, classes=[0, 1, 2])
	n_classes = y_test.shape[1]


		# print(predicted,y_test)
		# print(metrics.classification_report(y_test, predicted))
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
	plt.title('Receiver operating characteristic example')
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
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.savefig('roc_plot_dnn_classifier.png')
	# plt.show()
	# print(metrics.classification_report(y_test, predicted))


	

if __name__ == '__main__':
	main()

