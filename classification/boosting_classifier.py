from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups

def main():

	newsgroups_train = fetch_20newsgroups(subset='train')
	newsgroups_test = fetch_20newsgroups(subset='test')
	X_train = newsgroups_train.data
	X_test = newsgroups_test.data
	y_train = newsgroups_train.target
	y_test = newsgroups_test.target
	print(X_train[0])



	text_clf = Pipeline([('vect', CountVectorizer()),
	                     ('tfidf', TfidfTransformer()),
	                     # ('clf', GradientBoostingClassifier(n_estimators=100)),
	                     ])


	text_clf.fit(X_train, y_train)

	
	print(text_clf['vect'].transform(X_train).toarray())

	print(text_clf.transform(X_train).shape)
	# predicted = text_clf.predict(X_test)

	# print(metrics.classification_report(y_test, predicted))

if __name__ == '__main__':
	main()