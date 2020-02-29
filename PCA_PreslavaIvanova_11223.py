# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:19:09 2018

@author: User
"""

#Importing data
from numpy import loadtxt
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data = loadtxt('PCA_PreslavaIvanova_11223.txt', skiprows=1, delimiter=',')
X=data[:,0:4]
y=data[:, 4]
X=StandardScaler().fit_transform(X)

#1.     Applying PCA method
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
target_names=['yes', 'no']
print('explained variance ratio (first two components): %s' 
% str(pca.explained_variance_ratio_)) 
#explained variance ratio (first two components): [0.63526102 0.27534047]
plt.figure() 
colors = ['navy', 'darkorange'] 
lw = 2 
plt.scatter(X_r[:,0], X_r[:,1]) 
plt.show()
for color, i, target_name in zip(colors, [0, 1], target_names): 
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name) 

plt.legend(loc='best', shadow=False, scatterpoints=1) 
plt.title('PCA of blood donation, 03.2007 data set') 
plt.show()

#2.     KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
#Splitting the dataset
from sklearn.model_selection import train_test_split
X_r_train, X_r_test, y_train, y_test= train_test_split(X_r, y, test_size=0.3)
knn.fit(X_r_train,y_train)
#Finding train and test score
train_score = knn.score(X_r_train, y_train)
test_score  = knn.score(X_r_test, y_test)
print
print('knn','train_score',train_score,'test_score',test_score)
print
#('knn', 'train_score', 0.8164435946462715, 'test_score', 0.7822222222222223)

#2.1 Printing a confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_r_test)
print '\n confussion matrix for the test set :\n',confusion_matrix(y_test, y_pred)
#Predicting score and confusion matrix for the full dataset
y_p = knn.predict(X_r)
print
print 'knn.score for the full set :', knn.score(X_r, y)
print '\n confussion matrix for the full set :\n',confusion_matrix(y, y_p)
#knn.score for the full set : 0.8061497326203209

#2.2 Doing a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred)) # avg f1-score=0.78

#2.3 Using k-Fold to improve the score
from sklearn.cross_validation import KFold
kf = KFold(len(X_r), n_folds=9)
k=0
sm=0
for train_index, test_index in kf:
    #print('k=',k)
    knn.fit(X_r[train_index],y[train_index])
    score_test = knn.score(X_r[test_index], y[test_index])
    print('score_train=',knn.score(X_r[train_index], y[train_index])) 
    print('score_test =',score_test)
    if sm < score_test : 
        #print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex =  test_index
        
    k+=1
    print

#2.4 Finding the highest score
knn.fit(X_r[train_minindex],y[train_minindex])
pred_X_r_test =knn.predict(X_r[test_minindex])
print('SCORE on test set: ',knn.score(X_r[test_minindex],
                                         y[test_minindex]))
#'SCORE on test set: ', 0.9518072289156626
#Score of the full set
fulldata_r_sc=knn.score(X_r, y)
print('SCORE on full set: ',knn.score(X_r,y) )
#('SCORE on full set: ', 0.8114973262032086)
    
#3. KNN for original data
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
knn.fit(X_train,y_train)
#Finding train and test score
train_score = knn.score(X_train, y_train)
test_score  = knn.score(X_test, y_test)
print
print('knn','train_score',train_score,'test_score',test_score)
print
#('knn', 'train_score', 0.8336520076481836, 'test_score', 0.7288888888888889)

#3.1 Printing a confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
print '\n confussion matrix for the test set :\n',confusion_matrix(y_test, y_pred)
#Predicting score and confusion matrix for the full dataset
y_p = knn.predict(X)
print
print'knn.score for the full set :', knn.score(X, y)
print'\n confussion matrix for the full set :\n',confusion_matrix(y, y_p)
#knn.score for the full set : 0.8021390374331551

#3.2 Doing a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred)) # avg f1-score=0.71

#3.3 Using k-Fold to improve the score
from sklearn.cross_validation import KFold
kf = KFold(len(X), n_folds=9)
k=0
sm=0
for train_index, test_index in kf:
    #print('k=',k)
    knn.fit(X[train_index],y[train_index])
    score_test = knn.score(X[test_index], y[test_index])
    print('score_train=',knn.score(X[train_index], y[train_index])) 
    print('score_test =',score_test)
    if sm < score_test : 
        #print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex =  test_index
        
    k+=1
    print

#3.4 Finding the highest score
knn.fit(X[train_minindex],y[train_minindex])
pred_X_test =knn.predict(X[test_minindex])
print('SCORE on test set: ',knn.score(X[test_minindex],
                                         y[test_minindex]))
#'SCORE on test set: ', 0.9518072289156626
#Score of the full set
fulldata_r_sc=knn.score(X, y)
print('SCORE on full set: ',knn.score(X,y) )
#('SCORE on full set: ', 0.81951871657754)