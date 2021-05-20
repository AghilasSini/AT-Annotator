import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GMM
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier





parser = argparse.ArgumentParser(description='')
parser.add_argument('datafile', type=str, nargs=1, help='name of the input data file') 
args = parser.parse_args() 
dataFilename= args.datafile[0]

#Read the input dataset
mydata = pd.read_csv(dataFilename,sep=';')


#Preview input dataset
mydata.head()
print(mydata.shape)

#Plot the histograms
narr0 = mydata.loc[mydata['diag']=='narr']
dial = mydata.loc[mydata['diag']=='dial']

narr=narr0[:len(dial)]
frames=[narr,dial]
mydata=pd.concat(frames)

fig, axes = plt.subplots(9, 2, figsize=(9,17))

ax = axes.ravel()

print(ax.shape)

for i in range(1,16):
	ax[i].hist(narr.ix[:,i], bins='auto', color=mglearn.cm3(0), alpha=.5)
	ax[i].hist(dial.ix[:,i], bins='auto', color=mglearn.cm3(3), alpha=.5)
	ax[i].set_title(list(narr)[i])
	ax[i].set_yticks(())
    
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["", "dial"], loc="best")
fig.tight_layout()

#Prepare data for modeling
mydata.loc[mydata['diag']=='dial','label'] = 0
mydata.loc[mydata['diag']=='narr','label']=1
mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2)
scaler = StandardScaler()
scaler.fit(mydata_train.ix[:,1:16])
X_train = scaler.transform(mydata_train.ix[:,1:16])
X_test = scaler.transform(mydata_test.ix[:,1:16])
y_train = list(mydata_train['label'].values)
y_test = list(mydata_test['label'].values)

print(X_train.shape)
print(X_test.shape)


#Train decision tree model
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("Decision Tree")
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#Train random forest model
forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
print("Random Forests")
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

#Train gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print("Gradient Boosting")
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

#Train support vector machine model
svm = SVC().fit(X_train, y_train)
print("Support Vector Machine")
print("Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svm.score(X_test, y_test)))

#Train neural network model
mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
print("Multilayer Perceptron")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))


# Train linear regression model
print("Linear Regression")
regr = LinearRegression().fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(regr.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(regr.score(X_test, y_test)))


# Train GMM model
print("GMM")
n_classes = len(np.unique(y_train))
covar_type='full'
gmm=GMM(n_components=n_classes,covariance_type=covar_type, init_params='wc', n_iter=20).fit(X_train)
y_train_pred = gmm.predict(X_train)
train_accuracy = np.mean(y_train_pred.ravel() == y_train) # * 100
print("Accuracy on training set:{:.3f}".format(train_accuracy))
y_test_pred = gmm.predict(X_test)
test_accuracy = np.mean(y_test_pred.ravel() == y_test) #* 100
print("Accuracy on test set:{:.3f}".format(test_accuracy))

# Gaussian Naive Bayes model
print("Gaussian Naive Bayes")
gnb =GaussianNB().fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(gnb.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gnb.score(X_test, y_test)))


#Plot the variable importance
def plot_feature_importances_mydata(model):
	n_features = X_train.shape[1]
	plt.barh(range(n_features), model.feature_importances_, align='center')
	plt.yticks(np.arange(n_features), list(mydata))
	plt.xlabel("Variable importance")
	plt.ylabel("Independent Variable")

plot_feature_importances_mydata(tree)
plot_feature_importances_mydata(forest)
plot_feature_importances_mydata(gbrt)

#Plot the heatmap on first layer weights for neural network
plt.figure(figsize=(50, 10))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(1,10), list(mydata),fontsize = 10)
plt.xlabel("Columns in weight matrix", fontsize = 10)
plt.ylabel("Input feature", fontsize = 10)
plt.colorbar().set_label('Importance',size=10)
    
plt.show()
