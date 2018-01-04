#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
features_list = ['poi','salary' , 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

print "number of features to start with: ",len(features_list) - 1

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)
print "we create two new features here 'to_poi_message_ratio' and 'from_poi_message_ratio' "
for person in data_dict.values():
    person['to_poi_message_ratio'] = 0
    person['from_poi_message_ratio'] = 0
    if float(person['from_messages']) > 0:
        person['to_poi_message_ratio'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])
    if float(person['to_messages']) > 0:
        person['from_poi_message_ratio'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])
    
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
for iteration in range(40,80):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=iteration)
    print "Fitting the SVM classifier to the training set"
    t0 = time()
    param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
    # for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(features_train, labels_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by svm on :"
    print clf.best_estimator_
    print "Predicting "
    t0 = time()
    y_pred = clf.predict(features_test)
    print "done in %0.3fs" % (time() - t0)
    print "fl score: ",f1_score(labels_test, y_pred,average='weighted')
    print classification_report(labels_test, y_pred)
    print confusion_matrix(labels_test, y_pred)

    from sklearn.naive_bayes import GaussianNB
    print "Fitting the Decision  classifie to the training set"
    t0 = time()
    
    param_grid = {'criterion': ['gini', 'entropy'],'min_samples_split': [2, 10, 20],'max_depth': [None, 2, 5, 10],'min_samples_leaf': [1, 5, 10],
                 'max_leaf_nodes': [None, 5, 10, 20]}
    # for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
    clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
    clf = clf.fit(features_train, labels_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by svm on :"
    print clf.best_estimator_
    print "Predicting "
    t0 = time()
    y_pred = clf.predict(features_test)
    print "done in %0.3fs" % (time() - t0)
    print "fl score: ",f1_score(labels_test, y_pred,average='weighted')
    print classification_report(labels_test, y_pred)
    print confusion_matrix(labels_test, y_pred)

    print "Fitting the Decision  classifie to the training set"
    t0 = time()
    param_grid = {}
    # for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
    clf = GridSearchCV(GaussianNB(), param_grid)
    clf = clf.fit(features_train, labels_train)
    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by svm on :"
    print clf.best_estimator_
    print "Predicting "
    t0 = time()
    y_pred = clf.predict(features_test)
    print "done in %0.3fs" % (time() - t0)
    print "fl score: ",f1_score(labels_test, y_pred,average='weighted')
    print classification_report(labels_test, y_pred)
    print confusion_matrix(labels_test, y_pred)
    

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
