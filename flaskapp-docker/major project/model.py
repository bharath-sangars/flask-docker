import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

data = pd.read_csv("C:/7038/major project/heart.csv")

x=data.iloc[:,0:-1]
y=data.iloc[:,-1]

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)

sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

RF=RandomForestClassifier(n_estimators=100)
RF.fit(x_train,y_train)

ypred=RF.predict(x_test)
print("predicted values")
ypred

accuracy=accuracy_score(y_test,ypred)*100
print("Accuracy using random forest is : ",accuracy)

pipe = Pipeline([('scaler',sc),('RandomForest',RF)])
pipe.fit(x_train,y_train)

ypred=pipe.predict(x_test)
print("predicted values",ypred)


file_path = ("C:/7038/major project/RFmodel.pkl")
with open("C:/7038/major project/RFmodel.pkl", 'wb') as file:
    pickle.dump(pipe, file)

print(f"The object has been pickled and saved to {file_path}")



