import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('C:/Users/BLAUPLUG/Documents/Python_programs/Online payments fraud detection using ML/Online_payments_fraud.csv')
print(data.head())

#print(data.info())
#print(data.describe())
type_new = pd.get_dummies(data['type'], drop_first=True)
data_new = pd.concat([data, type_new], axis=1)
#print(data_new.head())

X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']

print(X.shape) 
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model=LogisticRegression()
model.fit(X_train, y_train)
	
	
train_preds = model.predict(X_train)
print('Training Accuracy : ', ras(y_train, train_preds))
	
y_preds = model.predict(X_test)
print('Validation Accuracy : ', ras(y_test, y_preds))
	



