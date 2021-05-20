import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC 
  
cancer = load_breast_cancer() 
  
# The data set is presented in a dictionary form: 
print(cancer.keys()) 
df_feat = pd.DataFrame(cancer['data'], 
                       columns = cancer['feature_names']) 
  
# cancer column is our target 
df_target = pd.DataFrame(cancer['target'],  
                     columns =['Cancer']) 
  
print("Feature Variables: ") 
print(df_feat.info()) 

print("Dataframe looks like : ") 
print(df_feat.head()) 


from sklearn.model_selection import train_test_split 
  
X_train, X_test, y_train, y_test = train_test_split( 
                        df_feat, np.ravel(df_target), 
                test_size = 0.30, random_state = 101) 


# train the model on train set 
model = SVC() 
model.fit(X_train, y_train) 
  
# print prediction results 
predictions = model.predict(X_test) 
print(classification_report(y_test, predictions)) 

from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','linear']
              }  
  
for cv in [5,6,7,8,9,10]:
	grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 1,cv=cv) 
	grid.fit(X_train, y_train)
	print(grid.best_params_) 
	print(grid.best_estimator_) 
	grid_predictions = grid.predict(X_test) 
	print(classification_report(y_test, grid_predictions)) 
# fitting the model for grid search 



# print best parameter after tuning 

  
# print how our model looks after hyper-parameter tuning 





# print classification report 