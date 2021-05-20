#Cluster-then-predict for classification tasks
#https://towardsdatascience.com/cluster-then-predict-for-classification-tasks-142fdfdc87d6
# author :Cole Brendel

from sklearn.datasets import make_classification


# Dataset
X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, n_classes=4)

import pandas as pd
df = pd.DataFrame(X, columns=['f{}'.format(i) for i in range(8)])


#Divide into Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=90210)



# Applying K-means

import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple
def get_clusters(X_train: pd.DataFrame, X_test: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	applies k-means clustering to training data to find clusters and predicts them for the test set
	"""
	clustering = KMeans(n_clusters=n_clusters, random_state=8675309,n_jobs=-1)
	clustering.fit(X_train)
	# apply the labels
	train_labels = clustering.labels_
	X_train_clstrs = X_train.copy()
	X_train_clstrs['clusters'] = train_labels
	
	# predict labels on the test set
	test_labels = clustering.predict(X_test)
	X_test_clstrs = X_test.copy()
	X_test_clstrs['clusters'] = test_labels
	return X_train_clstrs, X_test_clstrs
X_train_clstrs, X_test_clstrs = get_clusters(X_train, X_test, 2)


# Scaling


from sklearn.preprocessing import StandardScaler
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	applies standard scaler (z-scores) to training data and predicts z-scores for the test set
	"""
	scaler = StandardScaler()
	to_scale = [col for col in X_train.columns.values]
	scaler.fit(X_train[to_scale])
	X_train[to_scale] = scaler.transform(X_train[to_scale])
	
	# predict z-scores on the test set
	X_test[to_scale] = scaler.transform(X_test[to_scale])
	
	return X_train, X_test
X_train_scaled, X_test_scaled = scale_features(X_train_clstrs, X_test_clstrs)


# Experimentation

# to divide the df by cluster, we need to ensure we use the correct class labels, we'll use pandas to do that
train_clusters = X_train_scaled.copy()
test_clusters = X_test_scaled.copy()
train_clusters['y'] = y_train
test_clusters['y'] = y_test
# locate the "0" cluster
train_0 = train_clusters.loc[train_clusters.clusters < 0] # after scaling, 0 went negtive
test_0 = test_clusters.loc[test_clusters.clusters < 0]
y_train_0 = train_0.y.values
y_test_0 = test_0.y.values
# locate the "1" cluster
train_1 = train_clusters.loc[train_clusters.clusters > 0] # after scaling, 1 dropped slightly
test_1 = test_clusters.loc[test_clusters.clusters > 0]
y_train_1 = train_1.y.values
y_test_1 = test_1.y.values
# the base dataset has no "clusters" feature
X_train_base = X_train_scaled.drop(columns=['clusters'])
X_test_base = X_test_scaled.drop(columns=['clusters'])
# drop the targets from the training set
X_train_0 = train_0.drop(columns=['y'])
X_test_0 = test_0.drop(columns=['y'])
X_train_1 = train_1.drop(columns=['y'])
X_test_1 = test_1.drop(columns=['y'])
datasets = {
	'base': (X_train_base, y_train, X_test_base, y_test),
	'cluster-feature': (X_train_scaled, y_train, X_test_scaled, y_test),
	'cluster-0': (X_train_0, y_train_0, X_test_0, y_test_0),
	'cluster-1': (X_train_1, y_train_1, X_test_1, y_test_1),
}


from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
def run_exps(datasets: dict) -> pd.DataFrame:
	'''
	runs experiments on a dict of datasets
	'''
	# initialize a logistic regression classifier
	model = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=999, max_iter=250)
	
	dfs = []
	results = []
	conditions = []
	scoring = ['accuracy','precision_weighted','recall_weighted','f1_weighted']


for condition, splits in datasets.items():
	X_train = splits[0]
	y_train = splits[1]
	X_test = splits[2]
	y_test = splits[3]
	
	kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
	cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
	clf = model.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print(condition)
	print(classification_report(y_test, y_pred))
	results.append(cv_results)
	conditions.append(condition)
	this_df = pd.DataFrame(cv_results)
	this_df['condition'] = condition
	dfs.append(this_df)
	final = pd.concat(dfs, ignore_index=True)
	
	# We have wide format data, lets use pd.melt to fix this
	results_long = pd.melt(final,id_vars=['condition'],var_name='metrics', value_name='values')
	
	# fit time metrics, we don't need these
	time_metrics = ['fit_time','score_time'] 
	results = results_long[~results_long['metrics'].isin(time_metrics)] # get df without fit data
	results = results.sort_values(by='values')
	
	return results
df = run_exps(datasets)


# results

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20, 12))
sns.set(font_scale=2.5)
g = sns.boxplot(x="condition", y="values", hue="metrics", data=df, palette="Set3")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Dataset by Classification Metric')


pd.pivot_table(df, index='condition',columns=['metrics'],values=['values'], aggfunc='mean')





