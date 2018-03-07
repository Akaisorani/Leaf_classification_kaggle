#coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def warn(*args, **kwargs): pass
warnings.warn = warn



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
	
train, labels, test, test_ids, classes=encode(train,test)

sss=StratifiedShuffleSplit(labels,1,test_size=0.2,random_state=114514)

for train_index, test_index in sss:
	X_train, X_test=train.values[train_index], train.values[test_index]
	y_train, y_test = labels[train_index], labels[test_index]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
	
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
	clf.fit(X_train, y_train)
	name=clf.__class__.__name__
	
	print("="*30)
	print(name)
	
	print("****result****")
	train_predictions=clf.predict(X_test)
	acc=accuracy_score(y_test, train_predictions)
	print("Accuracy: {:.4%}".format(acc))
	result=np.eye(len(classes))[train_predictions]
	train_predictions=clf.predict_proba(X_test)
	ll = log_loss(y_test, train_predictions)
	ll2=log_loss(y_test, result)
	print("Log Loss: {}".format(ll))
	print("Log Loss2: {}".format(ll2))
	
	log_entry=pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
	log=log.append(log_entry)
	
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="g")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="b")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

