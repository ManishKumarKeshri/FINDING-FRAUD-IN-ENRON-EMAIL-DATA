#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL',0)
data = featureFormat(data_dict, features)

 
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
for key, value in data_dict.items():
    if value['bonus'] == data.max():
        print key
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue 
    
    outliers.append((key,int(val)))
    
outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:2])
for key in outliers_final:
    print key

big=0
for key, value in data_dict.items():
    if value['bonus'] >big:
        big=value['bonus']
        print key
big=0
for key, value in data_dict.items():
    if value['salary'] >big:
        big=value['salary']
        print key
#for key, value in data_dict.items():
 #   if value['bonus'] == big:
#        print key
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


