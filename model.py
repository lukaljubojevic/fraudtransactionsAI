import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

def fetch_data():
    client = MongoClient("mongodb://trening:lozinka123@192.168.59.103:32000/")  
    db = client["trening"]
    collection_name = "fraudData"
    collection = db[collection_name]
    cursor = collection.find({}, {'_id': 0}) 
    data = list(cursor)
    df = pd.DataFrame(data)
    return df

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

df=fetch_data()
print("XGBoost")
xgboost(df)