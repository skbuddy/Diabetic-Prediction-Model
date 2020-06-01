# Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

random.seed(123)


#Import dataset from .csv file
original_data = pd.read_csv('./mushrooms.csv')
#our Data File is Loaded into 'data'
#lets check our loaded Data By Following Command
data.head(4)


data.head()

habitat_dist = pd.value_counts(data['habitat'])
habitat_dist

class_dist = pd.value_counts(data['class'])
class_dist

#pie plotting
values = [habitat_dist[0],habitat_dist[1],habitat_dist[2],habitat_dist[3],habitat_dist[4],habitat_dist[5],habitat_dist[6]]
lb = ["Woods","Grass","Paths","Leaves","Urban","Meadows","Waste"]
plt.pie(values, labels=lb)
plt.title("Habitat Comparison")
plt.show()

#bar plot
plt.bar(["Edible","Poisonous"], [class_dist[0],class_dist[1]], color=['green', 'red'])
plt.xlabel("Types")
plt.ylabel("Percentage")
plt.title("Comparison of edible and poisonous mushrooms")
plt.show()


#data purification
#To Get Full Information Of the data
data.info()

# In this data We can find some null data like '?'
#'veil-type' and 'stalk-root' has Some NaN values 
#so Delete these 2 columns from the data
data.drop('veil-type',axis=1,inplace=True)
data.drop('stalk-root',axis=1,inplace=True)
data.info()


# String to numbers
#First Column Is Our Target
#It Has 2 Catagories ('p','e')
data['class'].unique()


#Label encoding
#Here we can see that all the columns of the dataframe are of the object 
#type so in order to properly analyze them, we need to encode the object 
#values in each column with the appropriate numerical value.
#In machine learning, we need to transform every data into integer of float 
#because in ML, only numbers are the valid ones.
#Our given data is in the form of strings
#So lets convert all our data in 'numbers' form

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in data.columns:
    data[i] = le.fit_transform(data[i])
data.head(5)

#Label and features
# We need features and labels to perform Our Further Process
# Our Target Is 'class' Column
Y = data['class']
# Except 'class' column all are features 
data1 = data
data1.drop('class',axis=1,inplace=True)

data1.head(5)

# we done with droping 'class' column from data
# So Remained Data is belongs to 'features'
#now load data1 into 'X'
X = data1
X.head(5)


#Model selection and predicting

#Using logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
print('Accuracy of logistic regression model is %.2f%%.' %(model.score(X_test,y_test)*100))

#Using support vector machine (SVM)
# Let's Take another model
from sklearn import svm
model = svm.SVC(kernel='rbf',gamma=0.3,C=1)
model.fit(X_train,y_train)
model.score(X_test,y_test)
print('Accuracy of SVM is %.2f%%.' %(model.score(X_test,y_test)*100))