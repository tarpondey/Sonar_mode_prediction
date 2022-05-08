import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Data collection and DAta processing
# loading the dataset to a pandas dataframe 
sonar_data=pd.read_csv('project1\sonar.all-data.csv', header=None)
# print(sonar_data.head())
print(sonar_data.shape)
# print(sonar_data.describe())
print(sonar_data[60].value_counts())

# print(sonar_data.groupby(60).mean())

# separating data and labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
# print(X)
# print(Y)

# Training and Test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)
# print(X_train)
# print(Y_train)
# MOdel Training->Logistic Regression

model=LogisticRegression()
# training the logistic Regression model with training data
model.fit(X_train,Y_train)

# model evaluation
# accuracy on the training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data:",training_data_accuracy)
# accuracy on test data   
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print("Accuracy on test data:",test_data_accuracy)

# Making a predictive system  
input_data=(0.0094,0.0333,0.0306,0.0376,0.1296,0.1795,0.1909,0.1692,0.1870,0.1725,0.2228,0.3106,0.4144,0.5157,0.5369,0.5107,0.6441,0.7326,0.8164,0.8856,0.9891,1.0000,0.8750,0.8631,0.9074,0.8674,0.7750,0.6600,0.5615,0.4016,0.2331,0.1164,0.1095,0.0431,0.0619,0.1956,0.2120,0.3242,0.4102,0.2939,0.1911,0.1702,0.1010,0.1512,0.1427,0.1097,0.1173,0.0972,0.0703,0.0281,0.0216,0.0153,0.0112,0.0241,0.0164,0.0055,0.0078,0.0055,0.0091,0.0067)

# changing input data to a numpy array  
input_data_as_numpy_array=np.asarray(input_data)
# reshape the numpy array as we are predicting for one instance 
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
# print(prediction)
if(prediction[0]=='R'):
    print("The object is a rock.")
else:
    print("The object is a mine")