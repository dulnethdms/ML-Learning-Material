#import libraries
import sys
import numpy
import matplotlib.pyplot
import pandas


#importing dataset
dataset = pandas.read_csv('Data.csv')
#print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(x)
#print(y)

#Taking Care of Missing data
from sklearn.impute import SimpleImputer #class
imputer = SimpleImputer(missing_values=numpy.nan, strategy="mean")
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
#print(x)

#Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0] )], remainder="passthrough")
x = ct.fit_transform(x)
#print (x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)

#Splitting training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)