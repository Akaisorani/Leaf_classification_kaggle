#coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')

def encode(train, test):
	le=LabelEncoder().fit(train.species)
	labels=le.transform(train.species)
	classes=list(le.classes_)
	test_ids=test.id
	train=train.drop(['species','id'],axis=1)
	test=test.drop(['id'],axis=1)
	
	return train, labels,test,test_ids,classes

def standard(train):
	Scaler=StandardScaler()
	Scaler.fit(train)
	train=Scaler.transform(train)
	return train
	
train, labels, test, test_ids, classes=encode(train,test)

def partition(train,labels):
	sss=StratifiedShuffleSplit(labels,1,test_size=0.2,random_state=None)
	for train_index, test_index in sss:
		X_mtrain, X_mtest=train.values[train_index], train.values[test_index]
		y_mtrain, y_mtest = labels[train_index], labels[test_index]
	return X_mtrain,y_mtrain,X_mtest,y_mtest

log_cols=["Classifier", "Accuracy", "Log Loss"]

# Predict Test Set
# clf = LogisticRegression(C=1e3,solver='lbfgs',tol=5e-4)
# clf=NuSVC(probability=True,tol=5e-4,nu=0.5)
# clf=LinearDiscriminantAnalysis()
# clf=KNeighborsClassifier(3)
clf=MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-3, hidden_layer_sizes=(), random_state=961218,tol=1e-8,verbose=False)

def testclf(clf,X_test,y_test):
	
	# clf.fit(X_train,y_train)
	# train_predi=clf.predict(X_test)
	# acc=accuracy_score(y_test,train_predi)
	X_test=standard(X_test)
	train_predi=clf.predict(X_test)
	acc=accuracy_score(y_test,train_predi)	
	print("Accuracy: {:.4%}".format(acc))
	train_predi_proba=clf.predict_proba(X_test)
	ll = log_loss(y_test, train_predi_proba)
	print("Log Loss: {}".format(ll))


sd_test=standard(test)
new_labels=None
clf.fit(standard(train),labels)
new_labels=clf.predict(sd_test)
new_proba=clf.predict_proba(sd_test)
# testclf(clf,sd_train,labels)
for i in range(5):
	X_mtrain,y_mtrain,X_mtest,y_mtest=partition(train,labels)
	
	if False : al_train=X_mtrain;al_labels=y_mtrain
	else:
		good_idxs=np.where(np.max(new_proba,axis=1)>0.99)
		# al_train=np.append(X_mtrain,test.values[good_idxs],axis=0)
		# al_labels=np.append(y_mtrain,new_labels[good_idxs],axis=0)
		al_train=np.append(train,test.values[good_idxs],axis=0)
		al_labels=np.append(labels,new_labels[good_idxs],axis=0)
		
		print(good_idxs[0].shape[0])

	al_train=standard(al_train)
	clf.fit(al_train,al_labels)
	testclf(clf,X_mtest,y_mtest)
	
	if True :old_labels=np.copy(new_labels)
	else: old_labels=None
	new_labels=clf.predict(sd_test)
	new_proba=clf.predict_proba(sd_test)
	if True :
		print('diff_num: ',np.sum(old_labels!=new_labels))
	



# al_train=np.append(train.values,test.values,axis=0)
# al_labels=np.append(labels,new_labels,axis=0)
# al_train=standard(al_train)
clf.fit(al_train, al_labels)
# test_predictions = clf.predict_proba(test)
test_predict=clf.predict(sd_test)
test_predict_proba=clf.predict_proba(sd_test)
# result=np.eye(len(classes))[test_predict]
result=test_predict_proba
# result[np.where(result>0.99)]=1
# result[np.where(result<0.01)]=0	
print(np.max(result,axis=1))
# Format DataFrame
submission = pd.DataFrame(result, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission2.csv', index = False)
#print(submission.tail())

