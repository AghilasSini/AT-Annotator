import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from sklearn.metrics import mean_squared_error


data = pd.read_csv('D://Blogs//insurance.csv')
testdata = pd.read_csv('D://Blogs//insuranceTest.csv')

mergedata = data.append(testdata)
testcount = len(testdata)
count = len(mergedata)-testcount



X_cat = mergedata.copy()
X_cat = mergedata.select_dtypes(include=['object'])
X_enc = X_cat.copy()



#ONEHOT ENCODING BLOCK


#X_enc = pd.get_dummies(X_enc, columns=['sex','region','smoker'])

#mergedata = mergedata.drop(['sex','region','smoker'],axis=1)


#END ENCODING BLOCK


# =============================================================================
# #LABEL ENCODING BLOCK
# 
X_enc = X_enc.apply(LabelEncoder().fit_transform) #
mergedata = mergedata.drop(X_cat.columns, axis=1)
# #END LABEL ENCODING BLOCK
# 
# =============================================================================


FinalData = pd.concat([mergedata,X_enc], axis=1)
train = FinalData[:count]
test = FinalData[count:]
trainy = train['charges'].astype('int')
trainx = train.drop(['charges'], axis=1)



test = test.drop(['charges'], axis=1)
X_train,X_test, y_train,y_test = train_test_split(trainx, trainy, test_size=0.3)


clf = xgboost.XGBRegressor()
clf.fit(X_train,y_train)
y_testpred= clf.predict(X_test)
y_pred = clf.predict(test)


dftestpred = pd.DataFrame(y_testpred)
dfpred = pd.DataFrame(y_pred)


rms = sqrt(mean_squared_error(y_test, y_testpred))


print("RMSE:", rms)