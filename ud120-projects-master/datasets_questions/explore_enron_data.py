#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

money,n_poi,n_sal,n_email,total_payments,n_poi_nan_pay=0,0,0,0,0,0
people = ("SKILLING JEFFREY K", "LAY KENNETH L","FASTOW ANDREW S") 
who = ""
print "no. of data points:",len(enron_data)
print "no. of features:",len(enron_data[enron_data.keys()[0]])
count=0
for i in enron_data:
    if(enron_data[i]["poi"]==1):
        n_poi+=1
        if enron_data[i]["total_payments"] == "NaN":
                n_poi_nan_pay += 1
    if enron_data[i]["email_address"] != "NaN":
	    n_email += 1
    if enron_data[i]["salary"] != "NaN":
	    n_sal += 1
    if enron_data[i]["total_payments"] == "NaN":
	    total_payments += 1
for i in people:
	if money<enron_data[i]["total_payments"]:
		money = enron_data[i]["total_payments"]
		who = i
print "count",count
print "value:",enron_data["PRENTICE JAMES"]["total_stock_value"]
[s for s in enron_data.keys() if "COLWELL" in s]
print enron_data['COLWELL WESLEY'],
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
[s for s in enron_data.keys() if "SKILLING" in s]
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print "- How many data points (people) are in the dataset?\n+ %r" % len(enron_data)
print "- For each person, how many features are available?\n+ %r" % len(enron_data["SKILLING JEFFREY K"]) 
print "- How many POIs are there in the E+F dataset?\n+ %r" % n_poi
# check the poi_names.txt file
print "- How many POIs were there total?\n+ %r" % 35
print "- What is the total value of the stock belonging to James Prentice?\n+ %r" % enron_data["PRENTICE JAMES"]["total_stock_value"]
print "- How many email messages do we have from Wesley Colwell to persons of interest?\n+ %r" % enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "- What is the value of stock options exercised by Jeffrey Skilling?\n+ %r" % enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print "- Of these three individuals (Lay, Skilling and Fastow), who took home the most money?\n+ %r, %r"% (who, money)
print "- How many folks in this dataset have a quantified salary?\n+ %r" % n_sal
print "- What about a known email address?\n+ %r" % n_email
print "- How many people in the E+F dataset have NaN for their total payments?\n+ %r" % total_payments
#(total_payments / float(len(enron_data)))
print "- What percentage of people in the dataset hane 'NaN' for their payments?\n+ %r" % (float(total_payments) / float(len(enron_data)))
print "- What percentage of POIs in the dataset hane 'NaN' for their payments?\n+ %r" % (float(n_poi_nan_pay) / float(n_poi))

poi_names = open("../final_project/poi_names.txt", "r")
fr=poi_names.readlines()
print "hjlk",len(fr[2:])
poi_names.close()
num_lines = sum(1 for line in open("../final_project/final_project_dataset.pkl"))
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
no_total_payments = len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))
print float(no_total_payments)/len(enron_data) * 100

print "NO of people:", len(enron_data)
print len(enron_data) + 10
print 10 + len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))

print "new poi:",n_poi+10
print "nan poi new",n_poi_nan_pay+10
#file_len('C:/Vindico/Projects/Code/Python/Python/Course/Udacity/Intro to Machine Learning/ud120-projects-master/final_project/poi_names.txt')
