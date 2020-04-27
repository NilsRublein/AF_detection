# -*- coding: utf-8 -*-
"""
University of Twente, Data Science 2019 1B, Project 6 [AF], Nils Rublein & Vera Dierx.

Sources:
    Handling imbalanced data:
    https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
    https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
    
    ROC and Precision vs Recall curve:
    https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

"""

import numpy as np
import pandas as pd
from pandas import read_excel
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import scipy.special as sp
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso
from sklearn import tree

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from DTW_KNN import KnnDtw
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.metrics import classification_report, confusion_matrix

#  ------------Load data -----------
my_sheet = 'Sheet1'
file_name = 'Preprocessed_AFData.xlsx' # name of your excel file
data = read_excel(file_name, sheet_name = my_sheet)

# Count classes and show them in a bar plot

counts = data.Control.value_counts()
sns.barplot(y = counts)
counts.plot(kind='bar', title='Number of Observations per class')
counts.plot.bar(x='Control',title='Number of Observations per class')
#percentage_AF_episodes = (len(data.loc[df.Control==1])) / (len(data.loc[df.Control == 0])) * 100


# Separate input features and target
y = data.Control
x = data.drop('Control', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)


# DummyClassifier to predict only target 0

dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# checking unique labels
print('Unique predicted labels: ', (np.unique(dummy_pred)))

# checking accuracy
print('Test score: ', accuracy_score(y_test, dummy_pred))
print('recall score: ', recall_score(y_test, dummy_pred))
print('precision score: ', precision_score(y_test, dummy_pred))


# Modeling the data as is
# Train model
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
 
# Predict on training set
lr_pred = lr.predict(X_test)
accuracy_score(y_test, lr_pred)
predictions = pd.DataFrame(lr_pred)
predictions[0].value_counts()
'''


# train a decision tree demodel

rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
# predict on test set
rfc_pred = rfc.predict(X_test)
'''

# ------------ Resampling ---------------

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_AF  = X[X.Control==0]
AF      = X[X.Control==1]


#Define a function for plotting ROC curves
def roc_plot (trainX, testX, trainy, testy, title='test'):
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    plt.figure()
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label=f'{title}')
    plt.legend()
    plt.show()
    
def precision_recall_plot(trainX, testX, trainy, testy, title='test'):
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(testX)
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(testy[testy==1]) / len(testy)
    plt.figure()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, label='Logistic')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label=f'{title}')
    plt.legend()
    plt.show()

def print_metrics(model):
   
    print('Accuracy score: ', accuracy_score(y_test, model))
    print('recall score: ', recall_score(y_test, model))
    print('precision score: ', precision_score(y_test, model))
    print('F1 score: ', f1_score(y_test, model))

#Oversample

AF_upsampled = resample(AF,
                          replace=True, # sample with replacement
                          n_samples=len(not_AF), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_AF, AF_upsampled])

# trying logistic regression again with the balanced dataset
y_train = upsampled.Control
X_train = upsampled.drop('Control', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)
upsampled_pred = upsampled.predict(X_test)

print_metrics(upsampled)
roc_plot(X_train, X_test, y_train, y_test, 'ROC curve for oversampling')
precision_recall_plot(X_train, X_test, y_train, y_test, 'Precision-recall curve for oversampling')


#Undersampling

not_AF_downsampled = resample(not_AF,
                                replace = False, # sample without replacement
                                n_samples = len(AF), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_AF_downsampled, AF])

y_train = downsampled.Control
X_train = downsampled.drop('Control', axis=1)

undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

undersampled_pred = undersampled.predict(X_test)

print_metrics(undersampled_pred)
roc_plot(X_train, X_test, y_train, y_test, 'ROC curve for undersampling')
precision_recall_plot(X_train, X_test, y_train, y_test, 'Precision-recall curve for undersampling')


#SMOTE

sm = SMOTE(random_state=27, ratio='minority')
X_train, y_train = sm.fit_sample(X_train, y_train)

smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)
smote_pred = smote.predict(X_test)

print_metrics(smote_pred)
roc_plot(X_train, X_test, y_train, y_test, 'ROC curve for SMOTE')
precision_recall_plot(X_train, X_test, y_train, y_test, 'Precision-recall curve for SMOTE')


#SMOTE + TOMEK (Oversampling followed by undersampling)

smt = SMOTETomek(random_state=27, ratio='minority')
X_train, y_train = smt.fit_sample(X_train, y_train)

smoteT = LogisticRegression(solver='liblinear').fit(X_train, y_train)
smoteT_pred = smoteT.predict(X_test)

print_metrics(smoteT_pred)
roc_plot(X_train, X_test, y_train, y_test, 'ROC curve for SMOTE + TOMEK')
precision_recall_plot(X_train, X_test, y_train, y_test, 'Precision-recall curve for SMOTE + TOMEK')

    
# ---------- Creating some more features -------------

# Convert the resampled X_train and y_train from np arrays back to dataframes
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

# Give X_train its column names back again
X_train.columns = X_test.columns.copy()

#Create a data frame that will contain all our features (for train and test respectively), including the ones we already have
x_train_features    = X_train.copy()
x_test_features     = X_test.copy()

# Mean
x_train_features["Mean"]  = X_train.mean(axis=1)
x_test_features["Mean"] = X_test.mean(axis=1)

# Standard Deviation (std)
x_train_features["Std"] = X_train.std(axis=1)
x_test_features["Std"] = X_test.std(axis=1)

# Root Mean Square of the Successive Differences (RMSSD)
diff_rri_train = np.diff(X_train)
diff_rri_test = np.diff(X_test)
x_train_features["RMSSD"] = np.sqrt(np.mean(diff_rri_train ** 2,axis=1))
x_test_features["RMSSD"] = np.sqrt(np.mean(diff_rri_test ** 2,axis=1))

# Min value
x_train_features["Min"] = X_train.min(axis=1)
x_test_features["Min"] = X_test.min(axis=1)

# Max value
x_train_features["Max"] = X_train.max(axis=1)
x_test_features["Max"] = X_test.max(axis=1)

#Median absolute deviation
x_train_features["Mad"] = X_train.mad(axis=1)
x_test_features["Mad"] = X_test.mad(axis=1)


# ------------ Feature Selection ---------------------
# Testing RFE, Stability Selection (Random Lasso) and Random forest feature importance to select best features


# Feature Importance from Random Forest
rfc = RandomForestClassifier(n_estimators=10).fit(x_train_features, y_train)
print ("Features sorted by their rank for the RandomForestClassifier :")
print (sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), x_train_features.columns), reverse=True))

importances = rfc.feature_importances_
indices = np.argsort(importances)
plt.figure()
plt.title('Random Forest Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), x_train_features.columns[indices])
plt.tick_params(axis ='y', labelsize = 7.5) 
plt.xlabel('Relative Importance')
plt.show()





# RFE
rfc2 = RandomForestClassifier()
rfe = RFE(rfc2, 20)
rfe = rfe.fit(x_train_features, y_train)
print(rfe.support_)
print(rfe.ranking_)
print ("Features sorted by their rank for RFE:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), x_train_features.columns)))



#Stability selection
rlasso = RandomizedLasso()
rlasso.fit(x_train_features, y_train)
 
print ("Features sorted by their score for Stability selection:")
print (sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
                 x_train_features.columns), reverse=True))

rlasso_scores = rlasso.scores_
indices = np.argsort(rlasso_scores)
plt.figure()
plt.title('Stability Selection Feature Importances')
plt.barh(range(len(indices)), rlasso_scores[indices], color='b', align='center')
plt.yticks(range(len(indices)), x_train_features.columns[indices])
plt.tick_params(axis ='y', labelsize = 7.5) 
plt.xlabel('Feature importance in percentages')
plt.show()


# Select new features
x_train_features_new    = x_train_features.copy().drop(['data1', 'Mean', 'data20','data21','data22','data23','data24','data25','data26','data27','data28','data29','data30'], axis=1)
x_test_features_new     = x_test_features.copy().drop(['data1', 'Mean', 'data20','data21','data22','data23','data24','data25','data26','data27','data28','data29','data30'], axis=1)

# -------- Classification Shizzle ---------------

def roc_plot_model(model_prob,testy,title='test', model_title ="test"):
    # generate a no skill prediction (majority class)
    ns_prob = [0 for _ in range(len(testy))]
    # keep probabilities for the positive outcome only
    model_prob = model_prob[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_prob)
    lr_auc = roc_auc_score(testy, model_prob)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_prob)
    lr_fpr, lr_tpr, _ = roc_curve(testy, model_prob)
    # plot the roc curve for the model
    plt.figure()
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, label=f'{model_title}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label=f'{title}')
    plt.legend()
    plt.show()
    
def precision_recall_plot_model(model_pred, model_prob, testy, title='test', model_title ="test" ):
    # keep probabilities for the positive outcome only
    model_prob = model_prob[:, 1]
    # predict class values
    lr_precision, lr_recall, _ = precision_recall_curve(testy, model_prob)
    lr_f1, lr_auc = f1_score(testy, model_pred), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(testy[testy==1]) / len(testy)
    plt.figure()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, label=f'{model_title}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label=f'{title}')
    plt.legend()
    plt.show()

#Logistic regression

#Logistic regression
def LR_several_features (feature_train, feature_test, ):    
    lr = LogisticRegression(solver='liblinear').fit(feature_train, y_train.values.ravel())
    lr_pred = lr.predict(feature_test)
    lr_prob = lr.predict_proba(feature_test)

    print_metrics(lr_pred)
    roc_plot_model(lr_prob, y_test.values.ravel(), 'ROC curve for lr')
    precision_recall_plot_model(lr_pred, lr_prob, y_test.values.ravel(), 'Precision-recall curve for lr')

#LR(x_train_features, x_test_features)

# K-nearest neighbours

m = KnnDtw(n_neighbors=1, max_warping_window=10)
m.fit(X_train.iloc[::100].values, y_train.iloc[::100].values)
label, proba = m.predict(X_test.iloc[::100].values)
print( classification_report(label, y_test.iloc[::100].values))

print('Accuracy score: ', accuracy_score(y_test.iloc[::100].values, label))
print('recall score: ', recall_score(y_test.iloc[::100].values, label))
print('precision score: ', precision_score(y_test.iloc[::100].values, label))
print('F1 score: ', f1_score(y_test.iloc[::100].values, label))

proba = np.column_stack((proba,label)) # Add predictions to probs for the plots
roc_plot_model(proba, y_test.iloc[::100].values.ravel(),title=f'ROC for KNN & DTW for original features.', model_title ="KNN & DTW")
precision_recall_plot_model(label, proba, y_test.iloc[::100].values.ravel(),title=f'Precision-Recall curve for KNN & DTW for original features.', model_title ="KNN & DTW" )


#SVM
#Naive Bayes classifiers

#Decision tree

def decisionTree(features_train_x, features_test_x, title='test' ):
    clf = tree.DecisionTreeClassifier().fit(features_train_x,y_train)
    clf_pred = clf.predict(features_test_x)
    clf_prob = clf.predict_proba(features_test_x)
    print_metrics(clf_pred)
    roc_plot_model(clf_prob, y_test ,title=f'ROC for decision tree for {title}.', model_title='Decision Tree')
    precision_recall_plot_model(clf_pred, clf_prob, y_test, title=f'Precision-Recall curve for decision tree for {title}.', model_title='Decision Tree')
    
#decisionTree(x_train_features_new, x_test_features_new, 'selected features')    
#decisionTree(x_train_features, x_test_features, 'all features') 
#decisionTree(X_train, X_test, 'original features') 



# All the data
clf2 = tree.DecisionTreeClassifier().fit(x_train_features,y_train)
clf_pred2 = clf2.predict(x_test_features)
print_metrics(clf_pred2)

# Check Just the 'raw' data
clf3 = tree.DecisionTreeClassifier().fit(X_train,y_train)
clf_pred3 = clf3.predict(X_test)
print_metrics(clf_pred3)



#Random Forests

def rand_forest(features_train_x, features_test_x, title='test' ):
    rfc = RandomForestClassifier(n_estimators=5000).fit(features_train_x, y_train)
    rfc_pred = rfc.predict(features_test_x)
    rfc_prob = rfc.predict_proba(features_test_x)
    print_metrics(rfc_pred)
    roc_plot_model(rfc_prob, y_test ,title=f'ROC for random forest for {title}.', model_title='Random Forest')
    precision_recall_plot_model(rfc_pred, rfc_prob, y_test, title=f'Precision-Recall curve for random forest for {title}.', model_title='Random Forest')
    
rand_forest(X_train, X_test, 'original features')


#Arima?

# ------------ Vera's code --------------
def LR_several_features (feature_train, feature_test, name1, name2, model_title):    
    lr = LogisticRegression(solver='liblinear').fit(feature_train, y_train.values.ravel())
    lr_pred = lr.predict(feature_test)
    lr_prob = lr.predict_proba(feature_test)
    print('Accuracy score: ', accuracy_score(y_test.values.ravel(), lr_pred))
    print('recall score: ', recall_score(y_test.values.ravel(), lr_pred))
    print('precision score: ', precision_score(y_test.values.ravel(), lr_pred))
    print('F1 score: ', f1_score(y_test.values.ravel(), lr_pred))

    roc_plot_model(lr_prob, y_test.values.ravel(), name1, model_title)
    precision_recall_plot_model(lr_pred, lr_prob, y_test.values.ravel(), name2, model_title)
   

#LR_several_features (x_train_features, x_test_features, 'ROC curve for Logistic Regression for all features.', 'Precision-Recall curve for Logistic Regression for all features.', 'Logistic Regression')
#LR_several_features(X_train, X_test, 'ROC curve for Logistic Regression for original features.', 'Precision-Recall curve for Logistic Regression for original features.', 'Logistic Regression')
LR_several_features (x_train_features_new, x_test_features_new, 'ROC curve for Logistic Regression for selected features.', 'Precision-Recall curve for Logistic Regression for selected features.', 'Logistic Regression')

#Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
def GNB (feature_train, feature_test, name1, name2, model_title):
    gnb = GaussianNB().fit(feature_train, y_train.values.ravel())
    gnb_pred = gnb.predict(feature_test)
    gnb_prob = gnb.predict_proba(feature_test)
    print('Accuracy score: ', accuracy_score(y_test.values.ravel(), gnb_pred))
    print('recall score: ', recall_score(y_test.values.ravel(), gnb_pred))
    print('precision score: ', precision_score(y_test.values.ravel(), gnb_pred))
    print('F1 score: ', f1_score(y_test.values.ravel(), gnb_pred))

    roc_plot_model(gnb_prob, y_test.values.ravel(), name1, model_title)
    precision_recall_plot_model(gnb_pred, gnb_prob, y_test.values.ravel(), name2, model_title)

GNB (x_train_features, x_test_features, 'ROC curve for Gaussian Naive Bayes Classifier for all features.', 'Precision-Recall curve for Gaussian Naive Bayes Classifier for all features.', 'Gaussian Naive Bayes Classifier')
#GNB (X_train, X_test, 'ROC curve for Gaussian Naive Bayes Classifier for original features.', 'Precision-Recall curve for Gaussian Naive Bayes Classifier for original features.', 'Gaussian Naive Bayes Classifier')
#GNB (x_train_features_new, x_test_features_new, 'ROC curve for Gaussian Naive Bayes Classifier for selected features.', 'Precision-Recall curve for Gaussian Naive Bayes Classifier for selected features.', 'Gaussian Naive Bayes Classifier')


from sklearn import svm
from sklearn.model_selection import GridSearchCV
def SVC2 (feature_train, feature_test, rows, name1, name2, model_title):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1, 100, 1000]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=5)
    grid_search.fit(feature_train.iloc [::rows], y_train.iloc [::rows].values.ravel())
    grid_search.best_params_
    print (grid_search.best_params_)

    svc=svm.SVC(kernel="poly", **grid_search.best_params_, probability=True)
    svc = svc.fit(feature_train.iloc [::rows], y_train.iloc [::rows].values.ravel())
    svc_pred =svc.predict(feature_test)
    svc_prob = svc.predict_proba(feature_test)
    print('Accuracy score: ', accuracy_score(y_test.values.ravel(), svc_pred))
    print('recall score: ', recall_score(y_test.values.ravel(), svc_pred))
    print('precision score: ', precision_score(y_test.values.ravel(), svc_pred))
    print('F1 score: ', f1_score(y_test.values.ravel(), svc_pred))

    roc_plot_model(svc_prob, y_test.values.ravel(), name1, model_title)
    precision_recall_plot_model(svc_pred, svc_prob, y_test.values.ravel(),  name2, model_title)
    

#SVC2 (x_train_features, x_test_features,500, 'ROC curve for Support Vector Machine for all features.', 'Precision-Recall curve for Support Vector Machine for all features.', 'Support Vector Machine')
#SVC2 (X_train, X_test,500, 'ROC curve for Support Vector Machine for original features.', 'Precision-Recall curve for Support Vector Machine for original features.', 'Support Vector Machine')
#SVC2 (x_train_features_new, x_test_features_new,500, 'ROC curve for Support Vector Machine for selected features.', 'Precision-Recall curve for Support Vector Machine for selected features.', 'Support Vector Machine')
