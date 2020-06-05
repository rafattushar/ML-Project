import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_excel("data.xlsx")
print(df.size)
#print(df.head())

sizes = df['Salary'].value_counts(sort = 1)
print(sizes)

#drop irrelevent column
#df.drop(['Capital-gain'], axis = 1, inplace = True)
#df.drop(['Capital-loss'], axis = 1, inplace = True)

#Handle Missing values
df = df.dropna()

print(df.size)

#convert non-numeric data to numeric
#method #1
df.Salary[df.Salary == ' <=50K'] = 0
df.Salary[df.Salary == ' >50K'] = 1


#print(df.dtypes)

#Method #2
df['WorkClass'] = df['WorkClass'].astype('category')
df['Education'] = df['Education'].astype('category')
df['Marital-status'] = df['Marital-status'].astype('category')
df['Occupation'] = df['Occupation'].astype('category')
df['Relationship'] = df['Relationship'].astype('category')
df['Race'] = df['Race'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Native-country'] = df['Native-country'].astype('category')
df['Salary'] = df['Salary'].astype('category')
#print(df.dtypes)

cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
#conversion of stirng to int in complete


#Define dependent variable or target class
Y = df['Salary'].values
#Y = Y.astype(int)

#Define independent variable or features
X = df.drop(labels = ['Salary'], axis = 1)

#Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 20)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, random_state = 30)

model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)

from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(Y_test, prediction_test))


feature_list = list(X.columns)
feature_importance = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending = False)
print(feature_importance)