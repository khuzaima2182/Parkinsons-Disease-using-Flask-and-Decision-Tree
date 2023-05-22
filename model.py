import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

df = pd.read_csv('Parkinsson disease.csv')
df.head()
df=df.drop(['name'],axis=1)
df.info()
df=df.drop_duplicates()
df.info()
corr=df.corr()
cor_target = abs(corr["status"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.3]
relevant_features
for x in df.columns:
    df[x]= (df[x]-df[x].min())/(df[x].max()-df[x].min())
df.head()
y=df['status']
x=df.drop(['status'],axis=1)
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
my_model_gini = DecisionTreeClassifier(criterion = "gini", max_depth = 5)
my_model_gini.fit(X_train, y_train)
y_pred_gini = my_model_gini.predict(X_test)
pickle.dump(my_model_gini, open("model.pkl", "wb"))

