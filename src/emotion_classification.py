import os
import sys


import pandas as pd
import numpy as np

import plotnine as p9

# Argument
import argparse

# Debug
import traceback
import time

__module_dir = os.path.dirname(__file__)


# from
# https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/
# adapted by Aghilas SINI <aghilas.sini@irisa.fr>


def main():
	parser = argparse.ArgumentParser(description="")
	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")
	# Add arguments
	parser.add_argument("inFilename", help="The input file to be projected")
	# Parsing arguments
	args = parser.parse_args()

	

	df = pd.read_csv(args.inFilename)

	convert_dict={
	'label_nominal':'category',
	'label_numeric':'category'
	}
	df = df.astype(convert_dict) 



	print(df.head())
	print(df.info())
	data_cols=["feature_{}".format(i) for i in range(1024)]
	target_cols=['label_nominal']
	X=df[data_cols]
	y=df[target_cols]

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


	# WITHOUT FEATURES NORMALISATION
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score

	model = RandomForestClassifier()
	model.fit(X_train, y_train)
	y_predict = model.predict(X_test)

	acc=accuracy_score(y_test.values, y_predict)
	print(acc)

	# features_dict = {}
	# Let’s show how important each feature was to helping our model perform.
	# for i in range(len(model.feature_importances_)):
		# features_dict[new_data_cols[i]] = model.feature_importances_[i]
	# sorted(features_dict.items(), key=lambda x:x[1], reverse=True)



	# # Feature Engineering

	# df['left_cross'] = df['left_distance'] * df['left_weight']
	# df['right_cross'] = df['right_distance'] * df['right_weight']


	# new_data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance', 'left_cross', 'right_cross']
	# new_target_cols = ['class_name']

	# X = df[new_data_cols]
	# y = df[new_target_cols]
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	# forest = RandomForestClassifier()
	# forest.fit(X_train, y_train)
	# y_predict = forest.predict(X_test)
	# acc2=accuracy_score(y_test, y_predict)
	# print(acc2)


	# new_data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance', 'left_cross', 'right_cross', 'left_right_ratio']
	# new_target_cols = ['class_name']

	# df['left_right_ratio'] = df['left_cross']/df['right_cross']
	# print(df.head())

	# X = df[new_data_cols]
	# y = df[new_target_cols]
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	# new_forest = RandomForestClassifier()
	# new_forest.fit(X_train, y_train)

	# y_predict = new_forest.predict(X_test)
	# acc3=accuracy_score(y_test, y_predict)
	# print(acc3)


	# # this is very important technique for our final data....


	# features_dict = {}
	# # Let’s show how important each feature was to helping our model perform.
	# for i in range(len(new_forest.feature_importances_)):
	#     features_dict[new_data_cols[i]] = new_forest.feature_importances_[i]
	# sorted(features_dict.items(), key=lambda x:x[1], reverse=True)

	# Hyperparameter Tuning With Grid Search
	# from sklearn.model_selection import GridSearchCV

	# gridsearch_forest = RandomForestClassifier()

	# params = {
	# 	"n_estimators": [100, 300, 500],
	# 	"max_depth": [5,8,15],
	# 	"min_samples_leaf" : [1, 2, 4]
	# }

	# clf = GridSearchCV(gridsearch_forest, param_grid=params, cv=5 )
	# clf.fit(X,y)


	# print(clf.best_params_)

	# import numpy as np
	# from sklearn.model_selection import KFold
	# # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
	# # y = np.array([1, 2, 3, 4])
	# ##
	# from sklearn.model_selection import train_test_split
	# # to split
	# from sklearn.model_selection import cross_val_score
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	# X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



	# kf = KFold(n_splits=5)
	# kf.get_n_splits(X_train)
	# print(kf)
	# i=0
	# for train_index, eval_index in kf.split(X_train):
	# 	print(i)
	# 	print("TRAIN:", train_index, "TEST:", eval_index)
	# 	X_train[]
	# 	i+=1
		# X_tr, X_eval = X_train[train_index], X_train[eval_index]
		# print(X_tr['filename'], X_eval['filename'])
		# y_train, y_eval = y[train_index], y[evak_index]


if __name__ == '__main__':
	main()
