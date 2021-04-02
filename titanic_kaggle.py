import os
from collections import Counter
from collections import defaultdict

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


titanic_test = pd.read_csv('test.csv')
titanic_train = pd.read_csv('train.csv')

titanic_train.head()

# Removing classifiers:
# Name (all unique)
# Cabin (mostly na)
# ticket: mostly unique
#


print(len(titanic_train))

embark_counts = defaultdict(int)

print(Counter(titanic_train['Embarked']))
print(Counter(titanic_train['Pclass']))
print(Counter(titanic_train['SibSp']))
print(Counter(titanic_train['Parch']))

nsolo = 0
npair = 0
nparen = 0
ngroup = 0

for ind, i in titanic_train.iterrows():
    if (i['SibSp'] == 0) & (i['Parch'] == 0):
        nsolo += 1
    elif (i['SibSp'] == 1) & (i['Parch'] == 0):
        npair += 1
    elif (i['SibSp'] == 0) & (i['Parch'] != 0):
        nparen += 1
    else:
        ngroup += 1

print([nsolo, npair, nparen, ngroup])

# tic3477 = titanic_train[titanic_train['Ticket'] == '347082']

# display(tic3477)
# print((titanic_train['Embarked'].unique()))

# lin_model = sk.linear_model.fit()

# Maintain original training set for posterity and comparison.

train_clean = titanic_train[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']].copy()



train_clean['Sex_M'] = titanic_train['Sex'] == 'male'


train_clean = train_clean.dropna(axis = 0)


train_clean.head()


sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(train_clean.drop('Survived',axis=1),
                                                    train_clean['Survived'], test_size = 0.30)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions_log = logmodel.predict(X_test)



rfmodel = RandomForestClassifier(n_estimators = 2000)
rfmodel.fit(X_train, y_train)
predictions_rf = rfmodel.predict(X_test)


svmmodel = svm.SVC()
svmmodel.fit(X_train, y_train)
predictions_svm = svmmodel.predict(X_test)

mlpc = MLPClassifier(hidden_layer_sizes=(5,5,5), max_iter = 250)
mlpc.fit(X_train, y_train)
predictions_mlpc = mlpc.predict(X_test)

X_pred = pd.DataFrame(X_test)
X_pred['predicted_log'] = predictions_log
X_pred['predicted_RF'] = predictions_rf
X_pred['predicted_SVM'] = predictions_svm
X_pred['predicted_MLPC'] = predictions_mlpc
X_pred['truth'] = y_test

print(type(y_test))

display(X_pred[(X_pred['predicted_log'] != X_pred['truth']) |
               (X_pred['predicted_RF'] != X_pred['truth']) |
               (X_pred['predicted_SVM'] != X_pred['truth']) |
               (X_pred['predicted_MLPC'] != X_pred['truth'])])


print(confusion_matrix(y_test,predictions_log))
print(confusion_matrix(y_test, predictions_rf))
print(confusion_matrix(y_test, predictions_svm))
print(confusion_matrix(y_test, predictions_mlpc))

X_true = train_clean.drop('Survived',axis=1)
y_true = train_clean['Survived']

test_clean = titanic_test[['Age', 'Fare', 'SibSp', 'Parch']].copy()


test_clean['Sex_M'] = titanic_test['Sex'] == 'male'
#test_clean = test_clean.dropna(axis = 0)
final_model = svm.SVC()

titanic_train.isna().sum()
titanic_test.isna().sum()


#final_model.fit(X_true, y_true)
#final = final_model.predict(test_clean)

#print(len(titanic_test))

#build the final model

final_model = RandomForestClassifier(n_estimators = 100, max_depth = 3)
final_model.fit(X_true,y_true)


survival = final_model.predict(test_data[['Sex_M', 'Fare', 'SibSp', 'Parch','Age']])

test_data['Survived'] = survival


Final_Predictions = test_data.drop(['Age', 'Fare', 'SibSp', 'Parch', 'Sex_M'],
                                   axis = 1).sort_index()

print(Final_Predictions)
Final_Predictions.to_csv('Final_Predictions.csv', index = False)