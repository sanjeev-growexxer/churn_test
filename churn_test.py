import pandas as pd
import numpy as np

pd.options.display.float_format = '{:.2f}'.format

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
#from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
import boto3

import pickle


data = pd.read_csv('creditcard.csv')
df1 = data[['V3','V4','V7','V10','V11','V12','V14','V16','V17','Class']].copy(deep = True)

over = SMOTE(sampling_strategy = 0.5)
under = RandomUnderSampler(sampling_strategy = 0.1)
f1 = df1.iloc[:,:9].values
t1 = df1.iloc[:,9].values

steps = [('under', under),('over', over)]
pipeline = Pipeline(steps=steps)
f1, t1 = pipeline.fit_resample(f1, t1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(f1, t1, test_size = 0.20, random_state = 2)

classifier = LogisticRegression(random_state = 0,C=10,penalty= 'l2') 
classifier.fit(x_train1,y_train1)
prediction = classifier.predict(x_test1)
cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train1,y_train1,cv = cv,scoring = 'roc_auc').mean()))
print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test1,prediction)))
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
    

s3 = boto3.client('s3')
file = "model.pkl"
bucket = "sanjeev-churn-pred-test"
s3_key = "model.pkl"

try:
    s3.upload_file(file, bucket, s3_key)
except Exception as e:
    print(e)
