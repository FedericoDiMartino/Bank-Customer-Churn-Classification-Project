# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 06:58:59 2020

@author: feder
"""
## Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


## Import data
churn_data = pd.read_csv("Churn_Modelling.csv", index_col = 0)

## Print head to show structure of data
print(churn_data.head())


## To work, all values need to be numeric
# churn_data.fillna(value=np.nan, inplace=True)
## reshape data so that geography column becomes three binary columns
heatmap_data = churn_data
heatmap_data['IsFrance'] = 0
heatmap_data['IsSpain'] = 0
heatmap_data['IsGermany'] = 0

heatmap_data.loc[heatmap_data['Geography'] == 'France','IsFrance'] = 1
heatmap_data.loc[heatmap_data['Geography'] == 'Spain','IsSpain'] = 1
heatmap_data.loc[heatmap_data['Geography'] == 'Germany','IsGermany'] = 1

heatmap_data['IsFrance'] = pd.to_numeric(heatmap_data['IsFrance'])
heatmap_data['IsSpain'] = pd.to_numeric(heatmap_data['IsSpain'])
heatmap_data['IsGermany'] = pd.to_numeric(heatmap_data['IsGermany'])

## Change gender column such that female -> 1, male -> 0
heatmap_data.loc[heatmap_data['Gender'] == 'Female','Gender'] = 1
heatmap_data.loc[heatmap_data['Gender'] == 'Male','Gender'] = 0
heatmap_data["Gender"] = pd.to_numeric(heatmap_data["Gender"])
#print(churn_data.head())


# Drop columns not be used
heatmap_data = heatmap_data.drop(['CustomerId', 'Surname', 'Geography'], axis = 'columns')

#print(heatmap_data.head())

#sns.heatmap(heatmap_data)
#plt.show()


# Calculate correlations
corr = heatmap_data.corr()

# Visualise correlation matrix
corr.style.background_gradient(cmap='coolwarm', axis = None).set_precision(2)

#corr.style.background_gradient(cmap='coolwarm', axis=None)




########

## Import libraries for next section
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

import time

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


## Set up data and target
X = heatmap_data.loc[:, heatmap_data.columns != 'Exited']
y = heatmap_data.iloc[:,9]

random.seed( 123456789 )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 123456789)


# First try a variety of methods and see which ones perform better
# Create list of methods to try

#from sklearn.cluster import KMeans

methods = [] # Generate empty list and then append name and function
methods.append(('KNN', KNeighborsClassifier()))
methods.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
methods.append(('DT', DecisionTreeClassifier()))
methods.append(('SVM', SVC(gamma='auto')))
methods.append(('RF', RandomForestClassifier(n_estimators = 100)))
methods.append(('BC', BaggingClassifier()))

# Function to build and test models from list of methods assigned
def model_tester(methods):
    results = []
    names = []
    for name, method in methods:
        start = time.time()
        end = time.time()
        kf = KFold(n_splits=10, random_state=123456789)
        cv_results = cross_val_score(method, X_train, y_train, cv= kf, scoring='accuracy')
        end = time.time()
        time_elapsed = end - start
        results.append(cv_results)
        
        names.append(name)
        print('%s Accuracy: Mean %f StD (%f) Time: %f' % (name, cv_results.mean(), cv_results.std(), time_elapsed))

model_tester(methods)
  
## Choose RF because high accuracy, not slowest
        
## Tune the RF hyperparameters
## What hyperparameters are there?
RF = RandomForestClassifier(n_estimators = 100,random_state=123456789) 
print(RF.get_params())

# Will tune n_estimators and max_depth,

# Total number of trees in the random forest
n_estimators = [int(i) for i in np.linspace(100, 100, num=10 ) ]  # has to be integers

# Maximum number of levels in each tree
max_depth = [int(i) for i in np.linspace(10, 90, num = 9) ]
max_depth = np.append(max_depth,None) ## 'None' means no arbitrary maximum




## Create hyperparameter grid
hyper_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth}

## instantiate grid search
grid_search = GridSearchCV(estimator = RF, param_grid = hyper_grid, 
                          cv = 5) # 5 fold

# fit to data
grid_search.fit(X_train, y_train)

## See which hyper-parameters are best
print(grid_search.best_params_)
# grid_search.best_estimator_

## Use best model on test set and let's see our results!

rf = RandomForestClassifier(n_estimators = 100, max_depth=10 ,random_state=123456789) 
rf.fit(X_train, y_train)
#print('Accuracy: {}'.format(rf.score(X_test , y_test)))

rf_predict = rf.predict(X_test)




print ( 'Accuracy:', accuracy_score(y_test, rf_predict))
print ('F1 score:', f1_score(y_test, rf_predict))
print ('Recall:', recall_score(y_test, rf_predict))
print ('Precision:', precision_score(y_test,rf_predict))
print ('\n clasification report:\n', classification_report(y_test, rf_predict))
print ('\n confussion matrix:\n',confusion_matrix(y_test, rf_predict))

# The recall and f1 score for case 1 (customer exiting) are surprisingly low.
# This is probably due to the data we have being unbalanced towards case 0
# However the macro and weighted averages are better, and these are less sensitive to class imbalance!

