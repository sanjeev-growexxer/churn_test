import boto3
import pandas as pd
import pickle
data= pd.read_csv('prediction_data.csv')
test_data =data[['V3','V4','V7','V10','V11','V12','V14','V16','V17']].copy(deep = True)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
preds = model.predict(test_data)
test_data['Class']=preds
test_data.to_csv('predicted_data.csv')
s3 = boto3.client('s3')
file = "predicted_data.csv"
bucket = "sanjeev-test-deploy"
s3_key = "predicted_data.csv"

try:
    s3.upload_file(file, bucket, s3_key)
except Exception as e:
    print(e)