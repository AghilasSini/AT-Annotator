import pandas as pd
# from
# https://ehackz.com/2018/03/23/python-scikit-learn-random-forest-classifier-tutorial/
# adapted by Aghilas SINI <aghilas.sini@irisa.fr>

column_names = ['class_name', 'left_weight', 'left_distance', 'right_weight', 'right_distance']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
header=None,
names=column_names
)
# print(df.head())
print(df.info())
print(df['class_name'].value_counts())



data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance']
target_cols = ['class_name']

X = df[data_cols]
y = df[target_cols]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

acc=accuracy_score(y_test.values, y_predict)
print(acc)


# Feature Engineering

df['left_cross'] = df['left_distance'] * df['left_weight']
df['right_cross'] = df['right_distance'] * df['right_weight']


new_data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance', 'left_cross', 'right_cross']
new_target_cols = ['class_name']

X = df[new_data_cols]
y = df[new_target_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_predict = forest.predict(X_test)
acc2=accuracy_score(y_test, y_predict)
print(acc2)


new_data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance', 'left_cross', 'right_cross', 'left_right_ratio']
new_target_cols = ['class_name']

df['left_right_ratio'] = df['left_cross']/df['right_cross']
print(df.head())

X = df[new_data_cols]
y = df[new_target_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

new_forest = RandomForestClassifier()
new_forest.fit(X_train, y_train)

y_predict = new_forest.predict(X_test)
acc3=accuracy_score(y_test, y_predict)
print(acc3)


# this is very important technique for our final data....


features_dict = {}
# Let’s show how important each feature was to helping our model perform.
for i in range(len(new_forest.feature_importances_)):
    features_dict[new_data_cols[i]] = new_forest.feature_importances_[i]
sorted(features_dict.items(), key=lambda x:x[1], reverse=True)

# Hyperparameter Tuning With Grid Search
from sklearn.model_selection import GridSearchCV

gridsearch_forest = RandomForestClassifier()

params = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5,8,15],
    "min_samples_leaf" : [1, 2, 4]
}

clf = GridSearchCV(gridsearch_forest, param_grid=params, cv=5 )
clf.fit(X,y)


print(clf.best_params_)
