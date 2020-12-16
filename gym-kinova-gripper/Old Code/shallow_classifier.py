import pandas as pd
import numpy as np
import matplotlib

#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

import pickle
from os import listdir
PATH = '/home/graspinglab/NCS_data/updated_data/'
all_files = listdir(PATH)
files = []
for f in all_files:
    if "pkl" in str(f):
        files.append(f)

X_train = []
X_test = []
y_train = []
y_test = []


for filename in files:
    file = open(PATH + filename, "rb")
    data = pickle.load(file)
    file.close()
    #80% of each file into training
    #20% of each file into testing
    X_train.extend(data["states"][:round(len(data["states"])*.8)])
    X_test.extend(data["states"][round(len(data["states"])*.8): len(data["states"])])

    y_train.extend(data["grasp_success"][:round(len(data["grasp_success"])*.8)])
    y_test.extend(data["grasp_success"][round(len(data["grasp_success"])*.8): len(data["grasp_success"])])

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)


from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#For saving and loading
from sklearn.externals import joblib

#Create the pipeline
pipeline = Pipeline([
    ('clf', LogisticRegression())
])

#Different of models to test
models = [#LogisticRegression(), 
          DecisionTreeClassifier(),
          #KNeighborsClassifier(),
          #GaussianNB(),
          #SVC(),
          RandomForestClassifier(n_estimators=20)]

#Cycle through the different models
for mod in models:
    
    pipeline.set_params(clf = mod)

    #Train and predict
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    
    print("_" * 3)
    print(str(mod))
    
    #Metrics
    print(accuracy_score(pred, y_test))
    print(confusion_matrix(pred, y_test))
    print(classification_report(pred, y_test))
    
    #Save
    joblib.dump(pipeline, PATH + type(mod).__name__ + '.joblib')
    #Load
    #clf = load('model.joblib') 
    #clf.predict(X_test)
    
    print("_" * 3)

