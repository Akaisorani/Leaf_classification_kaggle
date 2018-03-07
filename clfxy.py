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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

traino=pd.read_csv('./data/train.csv')
train=pd.read_csv('./xinche/huoche.csv',header=None)
test=pd.read_csv('./xinche/ceyan.csv',header=None)
labelin=pd.read_csv('./xinche/chepai.csv',header=None)

#标签转化
def encode(train, test,labelin,traino):
	le=LabelEncoder().fit(traino.species)
	classes=list(le.classes_)
	
	labels=labelin.values[:,0]
	test_ids=test[0]

	train=train.drop([0],axis=1)
	test=test.drop([0],axis=1)
	
	return train, labels,test,test_ids,classes

#标准化
def standard(train):
	Scaler=StandardScaler()
	Scaler.fit(train)
	train=Scaler.transform(train)
	return train
	
train, labels, test, test_ids, classes=encode(train,test,labelin,traino)

#随机划分训练集验证集
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
clf=MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-3, hidden_layer_sizes=(297,), random_state=961218,tol=1e-8,verbose=False)

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

bound=0.97
sd_test=standard(test)
new_labels=None
clf.fit(standard(train),labels)
new_labels=clf.predict(sd_test)
new_proba=clf.predict_proba(sd_test)
# testclf(clf,sd_train,labels)

#迭代更新训练集合测试集标签
for i in range(3):
	X_mtrain,y_mtrain,X_mtest,y_mtest=partition(train,labels)
	
	if False : al_train=X_mtrain;al_labels=y_mtrain
	else:
		#吸收优秀测试集数据
		good_idxs=np.where(np.max(new_proba,axis=1)>bound)
		#分验证集，用于测试
		# al_train=np.append(X_mtrain,test.values[good_idxs],axis=0)
		# al_labels=np.append(y_mtrain,new_labels[good_idxs],axis=0)
		#不分验证集，用于输出提交
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

#第二分类器
def secondclf(predict_proba,labels,classes,al_train,al_labels,test):
	def find_max_id(line,K):
		line=np.copy(line)
		maxlis=[]
		for i in range(K):
			id=np.argmax(line)
			maxlis.append(id)
			line[id]=-1
		return maxlis

	K=3
	# clf=RandomForestClassifier()
	# clf=NuSVC(probability=True,tol=5e-4,nu=0.5)
	clf = LogisticRegression(C=1e3,solver='lbfgs',tol=5e-4)
	# clf=SVC(kernel="rbf", C=1e3, probability=True)
	# clf=LinearDiscriminantAnalysis()
	
	stable_ids=np.where(np.max(predict_proba,axis=1)>bound)[0]
	unstable_ids=np.where(np.max(predict_proba,axis=1)<=bound)[0]
	res_stable=predict_proba[stable_ids]
	res_unstable=predict_proba[unstable_ids]
	for unstable_id in unstable_ids:
		line=predict_proba[unstable_id]
		max_ids=find_max_id(line,K)
		max_labels=list(map(lambda x:x+1,max_ids))
		usable_line_truths=al_labels==max_labels[0]
		for lab in max_labels:
			usable_line_truths=np.logical_or(usable_line_truths,al_labels==lab)
		
		sub_train=al_train[usable_line_truths]
		sub_labels=al_labels[usable_line_truths]
		sub_test=test[unstable_id].reshape(1,-1)
		clf.fit(sub_train,sub_labels)
		sub_predict_proba=clf.predict_proba(sub_test)
		# print("="*30)
		# print(unstable_id)
		# print(max_ids)
		# print(sub_predict_proba)
		# if sub_predict_proba[0][1]>0.047 and sub_predict_proba[0][1]<0.048:
			# sub_predict_proba[:,[0,1]]=sub_predict_proba[:,[1,0]]
		# print(sub_predict_proba)
		# print("="*30)
		max_ids.sort()
		tmp_proba=np.zeros((1,len(classes)))
		for i in range(len(max_ids)):
			tmp_proba[0][max_ids[i]]=sub_predict_proba[0][i]
		predict_proba[unstable_id]=tmp_proba
	
	return predict_proba
		
		



# al_train=np.append(train.values,test.values,axis=0)
# al_labels=np.append(labels,new_labels,axis=0)
# al_train=standard(al_train)
clf.fit(al_train, al_labels)
# test_predictions = clf.predict_proba(test)
test_predict=clf.predict(sd_test)
test_predict_proba=clf.predict_proba(sd_test)
# test_predict_proba=secondclf(test_predict_proba,test_predict,classes,al_train,al_labels,sd_test)

# result=np.eye(len(classes))[test_predict-1]
result=np.copy(test_predict_proba)
# print(result[np.logical_and(result>0.04,result<0.05)])

result[np.where(result>0.95)]=1
# result[range(result.shape[0]),np.argmax(result,axis=1)]=1
result[np.where(result<0.01)]=0
max_prob=np.max(result,axis=1)
print(max_prob)
# Format DataFrame
submission = pd.DataFrame(result, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

#预测结果分析
def analaysResult(pridict_proba,classes):
	pridict_proba=np.copy(pridict_proba)
	N=pridict_proba.shape[0]
	K=3
	
	res_id=np.zeros((N,K),dtype=np.int32)
	res_proba=np.zeros((N,K))
	res_label=[[] for i in range(K)]
	
	for i in range(K):
		res_id[:,i]=np.argmax(pridict_proba,axis=1)
		res_proba[:,i]=np.max(pridict_proba,axis=1)
		res_label[i]=[classes[res_id[x][i]] for x in range(N)]
		
		poses=[range(N),res_id[:,i]]
		pridict_proba[poses]=-1
	
	res_id=res_id+1
	res_proba[np.where(res_proba>0.99)]=1
	
	cols=['id',*['label_'+str(i) for i in range(K)],*['proba_'+str(i) for i in range(K)],*['name_'+str(i) for i in range(K)]]
	res=np.append(res_id,res_proba,axis=1)
	res=np.append(res,np.array(res_label).T,axis=1)
	res=np.append(test_ids.reshape(N,1),res,axis=1)
	
	sub_labels=pd.DataFrame(res,columns=cols)

	sub_labels.reset_index()
	sub_labels.to_csv('sub_labels.csv',index=False)
	

# Export Submission
submission.to_csv('submission_xy.csv', index = False)

#print(submission.tail())

analaysResult(test_predict_proba,classes)

