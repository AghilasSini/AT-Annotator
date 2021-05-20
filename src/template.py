
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Baseline
import sys
import codecs
import logging
import os
import re
from collections import defaultdict
from lxml import etree
from collections import OrderedDict


import pandas as pd

# Argument
import argparse
import numpy  as np

# Debug
import traceback
import time

__module_dir = os.path.dirname(__file__)



def main():
	parser = argparse.ArgumentParser(description="")

	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")

	# Add arguments
	parser.add_argument("path", help="The input file to be projected")

	# Parsing arguments
	args = parser.parse_args()


	data=pd.read_csv(args.path)
	data=data.fillna(0.0)
	
	convert_dict={
	'marker_label':'category',
	'marker_postag':'category', 
	'leftWord_postag':'category', 
	'lleftWord_postag':'category', 
	'rightWord_postag':'category',
	'rrightWord_postag':'category',
			
	}
	# # print(cat_list)
	data = data.astype(convert_dict) 



	data['marker_postag_id']=data.marker_postag.cat.codes
	data['leftWord_postag_id']=data.leftWord_postag.cat.codes
	data['lleftWord_postag_id']=data.lleftWord_postag.cat.codes
	data['rightWord_postag_id']=data.rightWord_postag.cat.codes
	data['rrightWord_postag_id']=data.rrightWord_postag.cat.codes
	data['label'] = data.marker_label.cat.codes
	print(data.marker_label)


	print(data.head(3))
	#"marker_postag_id",

	data_cols=["marker_duration","marker_energy_mean",'marker_f0_hz_mean','marker_f0_hz_mediane',
	"leftWord_postag_id","leftWord_duration","leftWord_energy_mean","leftWord_f0_hz_mean","leftWord_f0_hz_mediane",
	"lleftWord_postag_id","lleftWord_duration","lleftWord_energy_mean","lleftWord_f0_hz_mean","lleftWord_f0_hz_mediane",
	"rightWord_postag_id","rightWord_duration","rightWord_energy_mean","rightWord_f0_hz_mean","rightWord_f0_hz_mediane",
	"rrightWord_postag_id","rrightWord_duration","rrightWord_energy_mean","rrightWord_f0_hz_mean","rrightWord_f0_hz_mediane",
	'cur_brg_duration', 'cur_brg_f0_range', 'cur_brg_tone_range', 'cur_brg_artic_rate','cur_brg_nbr_wordlable', 'cur_brg_nbr_word',
	'right_brg_duration', 'right_brg_f0_range', 'right_brg_tone_range', 'right_brg_artic_rate','right_brg_nbr_wordlable', 'right_brg_nbr_word',
	'left_brg_duration', 'left_brg_f0_range', 'left_brg_tone_range', 'left_brg_artic_rate','left_brg_nbr_wordlable', 'left_brg_nbr_word',
	]
	target_cols=["label"]

	X = data[data_cols]
	y = data[target_cols]
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import confusion_matrix

	from sklearn.decomposition import PCA 
	from sklearn.preprocessing import StandardScaler
	


	model = RandomForestClassifier()
	model.fit(X_train, y_train)
	y_predict = model.predict(X_test)

	
	print(confusion_matrix(y_test.values, y_predict))


	acc=accuracy_score(y_test.values, y_predict)
	print(acc)

	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_samples, silhouette_score
	data_X=X
	PCA_model = PCA(n_components=3, random_state=42)
	X = PCA_model.fit_transform(X)*(-1)

	label_dict={0:"alors",1:"ensuite",2:"puis"}


	range_n_clusters =np.arange(2,3,+1)
	for n_clusters in range_n_clusters:
		clusterer = KMeans(n_clusters=n_clusters, random_state=42)
		cluster_labels = clusterer.fit_predict(X)
		silhouette_avg = silhouette_score(X, cluster_labels)
		print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)


	

if __name__ == '__main__':
	main()
