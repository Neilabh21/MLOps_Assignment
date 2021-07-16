import numpy as np
import pandas as pd

import dvc.api

import pickle

with dvc.api.open(
	'data/creditcard.csv',
	repo = "https://github.com/Neilabh21/MLOps_Assignment"
	) as fd:
	data = pd.read_csv(fd)

X = data.drop(['Class'], axis=1)
Y = data["Class"]

X_data = X.values
Y_data = Y.values

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

X_train.to_csv("../data/processed/train.csv", index=False)
X_test.to_csv("../data/processed/test.csv", index=False)

# Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier

# Create Decision Tree classifer object
clf = RandomForestClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# pkl file
pickle.dump(clf, open("../models/model.pkl", 'wb'))

from sklearn import metrics
import json
with open('../metrics/acc_f1.json', 'w') as f:
    json.dump(metrics.accuracy_score(y_test, y_pred), f)
    json.dump(metrics.f1_score(y_test, y_pred), f)




