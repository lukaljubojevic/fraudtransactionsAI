
import schedule
import time
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fetch_data():
    client = MongoClient("mongodb://trening:lozinka123@mongo-svc:27017/")
    db = client["trening"]
    collection_name = "fraudData"
    collection = db[collection_name]
    cursor = collection.find({}, {'_id': 0}) 
    data = list(cursor)
    df = pd.DataFrame(data)
    return df

def acc_report(actual,predicted):
    acc_score=accuracy_score(actual,predicted)
    class_rep=classification_report(actual,predicted)
    print('Preciznost modela:',acc_score)
    print(class_rep)

def xgboost(df):
    x=df.drop(['isFraud'],axis=1)
    y=df['isFraud']
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    x, y = smote.fit_resample(x, y)
    odabrani_stupci = ['type_TRANSFER', 'type_CASH_IN','type_CASH_OUT', 'amount', 'newbalanceOrig','newbalanceDest', 'isFraud']
    subset_data = df[odabrani_stupci]
    data = subset_data  
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = XGBClassifier(scale_pos_weight=3)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print(f"Preciznost: {accuracy}")
    print("\nMatrica konfuzije:")
    print(conf_matrix)
    print("\nIzvjestaj klasifikatora:")
    print(classification_rep)

def logreg(df):
    x=df.drop(['isFraud'],axis=1)
    y=df['isFraud']
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    x, y = smote.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    sc=StandardScaler()
    xtrain_sc = sc.fit_transform(x_train)
    xtest_sc=sc.transform(x_test)
    lgr = LogisticRegression(max_iter=500)
    lgr.fit(xtrain_sc, y_train)
    ypred_train_lgr=lgr.predict(xtrain_sc)
    ypred_test_lgr=lgr.predict(xtest_sc)
    print("Set trening podataka:")
    acc_report(y_train,ypred_train_lgr)
    print("Set testnih podataka:")
    acc_report(y_test,ypred_test_lgr)
    df=fetch_data()

def job():
   sys.stdout = open('results.txt', 'a')
   df=fetch_data()
   print("XGBoost")
   xgboost(df)
   print("Log. regresija")
   logreg(df)
   sys.stdout.close()
   sys.stdout = sys.__stdout__

schedule.every(5).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)