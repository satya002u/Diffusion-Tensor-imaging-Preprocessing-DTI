import numpy as np
import os
import re
#import pandas as pd
import pickle
from sklearn import svm
from numpy import *
#from sklearn.metrics import confusion_matrix
import math
from sklearn.feature_selection import SelectKBest, chi2,f_regression,f_classif,VarianceThreshold,RFECV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC
n_core = 15

def variance(train_data,test_data,train_label):
  variance = 0.95
  sel = VarianceThreshold(threshold=(variance * (1 - variance))).fit(train_data)
  train_data = sel.transform(train_data)
  test_data = sel.transform(test_data)
  # print ("after Variance Threshold of",variance,"data size : ", train_data.shape, test_data.shape)
  return train_data, test_data

def f_classify(train_data,test_data,train_label):
  kfeature =350
  if test_data.shape[1] < kfeature:
    if test_data.shape[1]>250:
      kfeature = 250
    elif test_data.shape[1]>200:
      kfeature = 200
    else:
      kfeature =100
  selector = SelectKBest(f_classif, k=kfeature)
  temp = selector.fit(train_data, train_label)
  train_data = temp.transform(train_data)
  test_data = temp.transform(test_data)
  # print ('f_classify selected dimensions ' , train_data.shape , test_data.shape)
    
  return train_data, test_data

def REF(train_data,test_data,train_label):
  fs_cv = 20
  estimator = SVC(kernel = "linear")
  selector1 = RFECV(estimator,step =1, cv=fs_cv, n_jobs = n_core)
  temp1 = selector1.fit(train_data, train_label)
  train_data = temp1.transform(train_data)
  test_data = temp1.transform(test_data)
  # print ("after REF with cv=",fs_cv,"Data size ",train_data.shape,test_data.shape)
  return train_data, test_data

def FeatureSelect(train_data, train_label, test_data,FS_type):
  # print ("======== Feature Selection Selected FS "+str(FS_type)+" ========")

  from sklearn.svm import SVC
  from sklearn.feature_selection import VarianceThreshold,chi2,RFECV
  #sel = VarianceThreshold().fit(train_data)

  if FS_type==1 : # VarianceThreshold only
    train_data,test_data = variance(train_data,test_data,train_label)

  elif FS_type == 2:# f_classif only
    train_data,test_data = f_classify(train_data,test_data,train_label)

  elif FS_type == 3: # REF only
    train_data,test_data = REF(train_data,test_data,train_label)

  elif FS_type == 4: #1+2
    train_data,test_data = variance(train_data,test_data,train_label)
    train_data,test_data = f_classify(train_data,test_data,train_label)

  elif FS_type == 5: #1+3
    train_data,test_data = variance(train_data,test_data,train_label)
    train_data,test_data = REF(train_data,test_data,train_label)

  elif FS_type == 6: #2+3
    train_data,test_data = f_classify(train_data,test_data,train_label)
    train_data,test_data = REF(train_data,test_data,train_label)
  return train_data,test_data



# def SVCC(train_data, train_label, test_data, test_label,classifier, target_names):
#   print ("======== Check Shape ========")
#   print ('Before Reshaping : ')
#   print ('data shape :', train_data.shape, train_label.shape, test_data.shape, test_label.shape)

#   from sklearn.svm import SVC,LinearSVC
#   from sklearn.model_selection import StratifiedKFold,KFold,permutation_test_score, GridSearchCV,cross_val_score,LeaveOneOut
#   #k_fold = LeaveOneOut()
#   k_fold = StratifiedKFold(20,shuffle=False)
#   print ("K_fold : ",k_fold)

#   '''FOR SVR '''
#   # parameter range
#   if classifier == 'svc':
#     clfer = SVC(decision_function_shape='ovo',class_weight="balanced")
#    # C_range = np.logspace(-1,15,10)
#     C_range = np.logspace(-2, 10, 13)
#    # gamma_range = np.logspace(-8, 5, 10)
#     gamma_range = np.logspace(-9, 3, 13)
#    # print "C range",C_range
#     #print "Gamma range",gamma_range
#     param_range = dict(gamma=gamma_range, C=C_range)
#    # param_range = {'kernel': ('rbf',), "C": [0.5,1,5.5,4,2,3.2] , "gamma": [0.0005,0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.5,0.0006]}  # epsilon max error allowed
#    # param_range = {'kernel': ('rbf',), "C": [5,5.1,5.15,4.9,4.8] , "gamma": [0.49,0.515,0.52,0.5,0.45]}  # epsilon max error allowed

#   elif classifier == 'nn':
#     from sklearn.neural_network import MLPClassifier
#     solver = np.array(['lbfgs', 'sgd', 'adam'])
#     alpha = np.array([0.01, 0.001,0.1])
#     learning_rate = np.array(['adaptive'])
#     momentum = np.array([0.9,0.8,0.5,0.7])
#     param_range = dict(solver = solver,alpha=alpha,learning_rate=learning_rate,momentum=momentum)
#     clfer = MLPClassifier()

#   elif classifier=="ada":
#     from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#     n_estimators = np.array([50,100,200,250,300,800])
#     learning_rate = np.array([0.1,1,0.5,0.8])
#     algorithm=np.array(['SAMME.R','SAMME'])
#     param_range=dict(n_estimators=n_estimators,learning_rate=learning_rate,algorithm=algorithm)
#     clfer=AdaBoostClassifier()

#   '''NN'''

#   def pval_cal(o_score,p_scores,p_times):
#     count = 0
#     for i in range (0,len(p_scores)):
#       if p_scores[i]>o_score:
#         count +=1
#     return (float(count+1))/(p_times+1)

#   clf = GridSearchCV(estimator=clfer, param_grid=param_range, cv=k_fold, n_jobs=n_core)
#   clf = clf.fit(train_data, train_label)
#   best_model =clf.best_estimator_
#   print ('-------Best Parameter ---------')
#   print ('best param ', clf.best_params_, "with score ",clf.best_score_)
#   predictTrain = best_model.predict(train_data)
#   predictTest = best_model.predict(test_data)
#   # target_names = ['AD', 'Normal']

#   print ("======== Train ========")
#   a= accuracy_score(train_label,predictTrain)
#   print ("accuracy : ", a)
#   print(classification_report(train_label, predictTrain, target_names=target_names))
#   print (confusion_matrix(train_label, predictTrain, labels=target_names))

#   print ("======== Test ========")
#   b = accuracy_score(test_label,predictTest)
#   cass_rep = classification_report(test_label, predictTest, target_names=target_names)
#   cm = confusion_matrix(test_label, predictTest, labels=target_names)
#   print ("accuracy : ",b)
#   print('classification_report test: ', cass_rep)
#   print ('confusion_matrix test :', cm)

'''
  print "======== Permutation Testing ========"
  print "-------- Train -------"
  score, permutation_scores, pvalue = permutation_test_score(best_model, train_data, train_label, scoring="accuracy", cv=k_fold, n_permutations=1000, n_jobs=n_core)
  print("Classification training %s (pvalue : %s)" % (score, pvalue))
  print "our p value ",pval_cal(a,permutation_scores,1000)

  print "-------- Test  -------"
  score, permutation_scores, pvalue = permutation_test_score(best_model, test_data, test_label, scoring="accuracy",cv=17, n_permutations=1000, n_jobs=n_core)
  print("Classification testing %s (pvalue : %s)" % (score, pvalue))
  print "our p value ", pval_cal(b, permutation_scores, 1000)
'''




# def initiator(subpath,fstype):
#   savingpath = main_path+subpath
#   train = np.load(savingpath+'train.npy')
#   test = np.load(savingpath+'test.npy')
#   train_grp = np.load(savingpath+'train_grp.npy')
#   test_grp = np.load(savingpath+'test_grp.npy')
#   print (savingpath)
#   train_data,test_data =FeatureSelect(train, train_grp, test,fstype)
#   SVCC(train_data, train_grp, test_data, test_grp,"svc")



# main_path = '/media/satya/win/3_QA/codesnew/NN_Vs_CSVM/data/wFA/reg_age_sex/'

# initiator('demean/',6)
