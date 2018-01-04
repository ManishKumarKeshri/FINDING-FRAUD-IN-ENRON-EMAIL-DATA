#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

n_poi=0
for i in predictions:
    n_poi+=i
print "total pois: ",n_poi

print "people in test dataset: ", len(X_test)


for i in range(len(predictions)):
    if(predictions[i]==1.0 and y_test[i]==1.0):
        print x
        print y
from sklearn import metrics
print "precison score: ",metrics.precision_score(y_test,predictions)
print "recall score: ",metrics.recall_score(y_test,predictions)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print "precison score: ",metrics.precision_score(true_labels,predictions)
print "recall score: ",metrics.recall_score(true_labels,predictions)

true_post=0
for i in range(len(predictions)):
    if(true_labels[i]==1 and predictions[i]==1):
        true_post+=1
print "True postives: ",true_post

true_neg=0
for i in range(len(predictions)):
    if(true_labels[i]==0 and predictions[i]==0):
        true_neg+=1
print "True neg: ",true_neg

false_pos=0
for i in range(len(predictions)):
    if(true_labels[i]==0 and predictions[i]==1):
        false_pos+=1
print "False pos: ",false_pos

false_neg=0
for i in range(len(predictions)):
    if(true_labels[i]==1 and predictions[i]==0):
        false_neg+=1
print "True neg: ",false_neg


