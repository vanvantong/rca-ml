# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, classification_report,accuracy_score, f1_score
from datetime import datetime
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Conv2D, Flatten, AveragePooling2D
from keras.layers import Permute
import time
import datetime
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from itertools import combinations 


from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier


def classifaction_report_csv(report,precision,recall,f1_score,fold):
	"""Generate the report to data processing"""
	with open('classification_report_diagnosis_gradient_boosting_100estimators.csv', 'a') as f:
		report_data = []
		lines = report.split('\n')
		row = {}
		row['class'] =  "fold %u" % (fold+1)
		report_data.append(row)
		for line in lines[2:44]:
			row = {}
			line = " ".join(line.split())
			row_data = line.split(' ')
			#print row_data
			if(len(row_data)>2):
				if(row_data[1]!='avg'):
					row['class'] = row_data[0]
					row['precision'] = float(row_data[1])
					row['recall'] = float(row_data[2])
					row['f1_score'] = float(row_data[3])
					row['support'] = row_data[4]
					report_data.append(row)
				else:
					row['class'] = row_data[0]+row_data[1]
					row['precision'] = float(row_data[2])
					row['recall'] = float(row_data[3])
					row['f1_score'] = float(row_data[4])
					row['support'] = row_data[5]
					report_data.append(row)
		row = {}
		row['class'] = 'macro'
		row['precision'] = float(precision)
		row['recall'] = float(recall)
		row['f1_score'] = float(f1_score)
		row['support'] = 0
		report_data.append(row)
		dataframe = pd.DataFrame.from_dict(report_data)
		dataframe.to_csv(f, index = False)

time_step = 10
feature = 9

X = []
y = []
arrLabel = []

#Processing data
f = open('Dataset_Static_Network.csv','r')
for line in f:
	arrPara = []
	X_tmp = []

	tmp = line.split(';')
	if len(tmp) == 10:
		for j in range(9):
			s = tmp[j].split(',')
			if len(s) == 10:
				for i in range(10):
					arrPara.append(float(s[i]))

		arrLabel.append([float(tmp[9])])

		
		X_tmp = [arrPara[i] for i in range(len(arrPara))]
		X_tmp = np.array(X_tmp)
		X.append(X_tmp)
		

X = np.array(X)
y = [arrLabel[i] for i in range(len(arrLabel))]
y = np.array(y)

fold = 1

sss1 = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
for train, test in sss1.split(X, y):
	X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

'''
#SVM
svclassifier = SVC(kernel='linear')
start_training = time.time()
svclassifier.fit(X_train, y_train)
training_time = time.time() - start_training

start_testing = time.time()
y_pred = svclassifier.predict(X_test)
testing_time = time.time() - start_testing
'''

"""
#Tune parameters for random forest 
# number of trees in random forest

rfc = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}# Random search of parameters

rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the model
rfc_random.fit(X_train, y_train)
print(rfc_random.best_params_)

"""

'''
# random forest 
#{'n_estimators': 1400, 'max_features': 'auto', 'max_depth': 100}
rfc = RandomForestClassifier(n_estimators = 100, max_depth=100, max_features='auto')
#rfc = RandomForestClassifier(n_estimators=1400, max_depth=100, max_features='auto')
#start_training = time.time()
rfc.fit(X_train,y_train)
#training_time = time.time() - start_training

#start_testing = time.time()
y_pred = rfc.predict(X_test)
#testing_time = time.time() - start_testing
'''

'''
#Adaboost
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
start_training = time.time()
clf.fit(X_train,y_train)
training_time = time.time() - start_training

start_testing = time.time()
y_pred = clf.predict(X_test)
testing_time = time.time() - start_testing
'''

'''
#Bagging
clf = BaggingClassifier(base_estimator=SVC(),n_estimators=100, random_state=0)
start_training = time.time()
clf.fit(X_train, y_train)
training_time = time.time() - start_training

start_testing = time.time()
y_pred = clf.predict(X_test)
testing_time = time.time() - start_testing
'''

#Gradient Boosting
n_estimator = 100
clf = GradientBoostingClassifier(n_estimators=n_estimator)
start_training = time.time()
clf.fit(X_train, y_train)
training_time = time.time() - start_training

start_testing = time.time()
y_pred = clf.predict(X_test)
testing_time = time.time() - start_testing



#f = open("TimeComplexity.csv", "a")
#f.write("Gradient Boosting 100estimators, Training:" + str(training_time) + ", Testing:" + str(testing_time)+", Size of training:" + str(len(X_train)) + ", Size of testing:"+str(len(X_test))+"\n")
#f.close()


y_result = y_pred
score = f1_score(y_test, y_result,average="macro")
precision = precision_score(y_test, y_result,average="macro")
recall = recall_score(y_test, y_result,average="macro")
report = classification_report(y_test,y_result,digits=4)
acc= accuracy_score(y_test, y_result)
print '\n clasification report:\n', report
print 'F1 score:', score
print 'Recall:', recall
print 'Precision:', precision
print 'Acc:', acc
classifaction_report_csv(report,precision,recall,score,fold)	
		
print '\nFinish\n'				
